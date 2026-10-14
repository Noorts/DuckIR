#include "index/vss_blob_functions.hpp"
#include "index/vss_blob_codec.hpp"
#include "index/vss_module.hpp"

#include "duckdb/common/types/blob.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <cmath>

namespace duckdb {

//------------------------------------------------------------------------------
// Distance Kernels (mirroring core_functions/array_kernels.hpp)
//------------------------------------------------------------------------------

struct BlobDistanceOp {
	template <class TYPE>
	static TYPE Operation(const TYPE *lhs, const TYPE *rhs, idx_t count) {
		TYPE dist = 0;
		for (idx_t i = 0; i < count; i++) {
			auto diff = lhs[i] - rhs[i];
			dist += diff * diff;
		}
		return std::sqrt(dist);
	}
};

struct BlobCosineDistanceOp {
	template <class TYPE>
	static TYPE Operation(const TYPE *lhs, const TYPE *rhs, idx_t count) {
		TYPE dot = 0, norm_l = 0, norm_r = 0;
		for (idx_t i = 0; i < count; i++) {
			dot += lhs[i] * rhs[i];
			norm_l += lhs[i] * lhs[i];
			norm_r += rhs[i] * rhs[i];
		}
		auto similarity = dot / std::sqrt(norm_l * norm_r);
		similarity = std::max(static_cast<TYPE>(-1.0), std::min(similarity, static_cast<TYPE>(1.0)));
		return static_cast<TYPE>(1.0) - similarity;
	}
};

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------

static void EncodeBlobFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto &input = args.data[0];

	UnifiedVectorFormat input_format;
	input.ToUnifiedFormat(count, input_format);

	auto &child_vec = ListVector::GetEntry(input);
	auto child_data = FlatVector::GetData<float>(child_vec);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(input_format);

	for (idx_t i = 0; i < count; i++) {
		auto idx = input_format.sel->get_index(i);
		if (!input_format.validity.RowIsValid(idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto &entry = list_entries[idx];
		auto blob_size = entry.length * sizeof(int16_t);
		auto blob = StringVector::EmptyString(result, blob_size);
		EncodeFloatArrayToBlob(child_data + entry.offset, entry.length, data_ptr_cast(blob.GetDataWriteable()));
		blob.Finalize();
		FlatVector::GetData<string_t>(result)[i] = blob;
	}

	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void DecodeBlobFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto &input = args.data[0];

	UnifiedVectorFormat input_format;
	input.ToUnifiedFormat(count, input_format);

	auto input_data = UnifiedVectorFormat::GetData<string_t>(input_format);

	auto &child_vec = ListVector::GetEntry(result);
	idx_t total_elements = 0;

	// First pass: count total elements
	for (idx_t i = 0; i < count; i++) {
		auto idx = input_format.sel->get_index(i);
		if (input_format.validity.RowIsValid(idx)) {
			total_elements += BlobDimensionCount(input_data[idx].GetSize());
		}
	}

	ListVector::Reserve(result, total_elements);
	auto child_data = FlatVector::GetData<float>(child_vec);
	auto list_entries = FlatVector::GetData<list_entry_t>(result);

	idx_t offset = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = input_format.sel->get_index(i);
		if (!input_format.validity.RowIsValid(idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto &blob = input_data[idx];
		auto dim_count = BlobDimensionCount(blob.GetSize());
		list_entries[i].offset = offset;
		list_entries[i].length = dim_count;
		DecodeBlobToFloatArray(const_data_ptr_cast(blob.GetData()), blob.GetSize(), child_data + offset);
		offset += dim_count;
	}
	ListVector::SetListSize(result, offset);

	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void BlobToBase64Function(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t blob) {
		auto base64_size = Blob::ToBase64Size(blob);
		auto base64 = StringVector::EmptyString(result, base64_size);
		Blob::ToBase64(blob, base64.GetDataWriteable());
		base64.Finalize();
		return base64;
	});
}

static void Base64ToBlobFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t str) {
		auto blob_size = Blob::FromBase64Size(str);
		auto blob = StringVector::EmptyString(result, blob_size);
		Blob::FromBase64(str, data_ptr_cast(blob.GetDataWriteable()), blob_size);
		blob.Finalize();
		return blob;
	});
}

//------------------------------------------------------------------------------
// Distance Function Overloads (BLOB, ARRAY)
//------------------------------------------------------------------------------

struct BlobDistanceBindData : public FunctionData {
	idx_t array_size;

	explicit BlobDistanceBindData(idx_t array_size) : array_size(array_size) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<BlobDistanceBindData>(array_size);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<BlobDistanceBindData>();
		return array_size == other.array_size;
	}
};

// Bind for (BLOB, ARRAY) - blob is arg 0, array is arg 1
static unique_ptr<FunctionData> BlobArrayDistanceBind(ClientContext &context, ScalarFunction &bound_function,
                                                      vector<unique_ptr<Expression>> &arguments) {
	auto &array_type = arguments[1]->return_type;
	if (array_type.id() != LogicalTypeId::ARRAY) {
		throw BinderException("%s: second argument must be an ARRAY type", bound_function.name);
	}
	auto array_size = ArrayType::GetSize(array_type);
	bound_function.arguments[1] = LogicalType::ARRAY(LogicalType::FLOAT, array_size);
	return make_uniq<BlobDistanceBindData>(array_size);
}

// Bind for (ARRAY, BLOB) - array is arg 0, blob is arg 1
static unique_ptr<FunctionData> ArrayBlobDistanceBind(ClientContext &context, ScalarFunction &bound_function,
                                                      vector<unique_ptr<Expression>> &arguments) {
	auto &array_type = arguments[0]->return_type;
	if (array_type.id() != LogicalTypeId::ARRAY) {
		throw BinderException("%s: first argument must be an ARRAY type", bound_function.name);
	}
	auto array_size = ArrayType::GetSize(array_type);
	bound_function.arguments[0] = LogicalType::ARRAY(LogicalType::FLOAT, array_size);
	return make_uniq<BlobDistanceBindData>(array_size);
}

// Execute: (BLOB, ARRAY<FLOAT>[N]) → FLOAT
template <class OP>
static void BlobArrayDistanceExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_state = state.Cast<ExecuteFunctionState>();
	auto &bind_data = func_state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<BlobDistanceBindData>();
	auto array_size = bind_data.array_size;

	auto count = args.size();
	auto &blob_vec = args.data[0];
	auto &array_vec = args.data[1];

	UnifiedVectorFormat blob_format;
	blob_vec.ToUnifiedFormat(count, blob_format);
	auto blob_data = UnifiedVectorFormat::GetData<string_t>(blob_format);

	UnifiedVectorFormat array_format;
	array_vec.ToUnifiedFormat(count, array_format);

	auto &array_child = ArrayVector::GetEntry(array_vec);
	auto array_child_data = FlatVector::GetData<float>(array_child);

	auto res_data = FlatVector::GetData<float>(result);

	auto decoded = make_unsafe_uniq_array<float>(array_size);

	for (idx_t i = 0; i < count; i++) {
		auto blob_idx = blob_format.sel->get_index(i);
		auto array_idx = array_format.sel->get_index(i);

		if (!blob_format.validity.RowIsValid(blob_idx) || !array_format.validity.RowIsValid(array_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto &blob = blob_data[blob_idx];
		if (BlobDimensionCount(blob.GetSize()) != array_size) {
			throw InvalidInputException("BLOB dimension count (%llu) does not match array size (%llu)",
			                            BlobDimensionCount(blob.GetSize()), array_size);
		}

		DecodeBlobToFloatArray(const_data_ptr_cast(blob.GetData()), blob.GetSize(), decoded.get());
		res_data[i] =
		    OP::template Operation<float>(decoded.get(), array_child_data + array_idx * array_size, array_size);
	}

	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// Execute: (ARRAY<FLOAT>[N], BLOB) → FLOAT
template <class OP>
static void ArrayBlobDistanceExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_state = state.Cast<ExecuteFunctionState>();
	auto &bind_data = func_state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<BlobDistanceBindData>();
	auto array_size = bind_data.array_size;

	auto count = args.size();
	auto &array_vec = args.data[0];
	auto &blob_vec = args.data[1];

	UnifiedVectorFormat array_format;
	array_vec.ToUnifiedFormat(count, array_format);
	auto &array_child = ArrayVector::GetEntry(array_vec);
	auto array_child_data = FlatVector::GetData<float>(array_child);

	UnifiedVectorFormat blob_format;
	blob_vec.ToUnifiedFormat(count, blob_format);
	auto blob_data = UnifiedVectorFormat::GetData<string_t>(blob_format);

	auto res_data = FlatVector::GetData<float>(result);

	auto decoded = make_unsafe_uniq_array<float>(array_size);

	for (idx_t i = 0; i < count; i++) {
		auto array_idx = array_format.sel->get_index(i);
		auto blob_idx = blob_format.sel->get_index(i);

		if (!array_format.validity.RowIsValid(array_idx) || !blob_format.validity.RowIsValid(blob_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto &blob = blob_data[blob_idx];
		if (BlobDimensionCount(blob.GetSize()) != array_size) {
			throw InvalidInputException("BLOB dimension count (%llu) does not match array size (%llu)",
			                            BlobDimensionCount(blob.GetSize()), array_size);
		}

		DecodeBlobToFloatArray(const_data_ptr_cast(blob.GetData()), blob.GetSize(), decoded.get());
		res_data[i] =
		    OP::template Operation<float>(array_child_data + array_idx * array_size, decoded.get(), array_size);
	}

	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

template <class OP>
static void RegisterDistanceOverloads(ExtensionLoader &loader, const string &func_name) {
	auto array_type = LogicalType::ARRAY(LogicalType::FLOAT, optional_idx());

	// (BLOB, ARRAY<FLOAT>[N]) → FLOAT
	ScalarFunction blob_array(func_name, {LogicalType::BLOB, array_type}, LogicalType::FLOAT,
	                          BlobArrayDistanceExecute<OP>, BlobArrayDistanceBind);
	loader.AddFunctionOverload(blob_array);

	// (ARRAY<FLOAT>[N], BLOB) → FLOAT
	ScalarFunction array_blob(func_name, {array_type, LogicalType::BLOB}, LogicalType::FLOAT,
	                          ArrayBlobDistanceExecute<OP>, ArrayBlobDistanceBind);
	loader.AddFunctionOverload(array_blob);
}

//------------------------------------------------------------------------------
// Registration
//------------------------------------------------------------------------------

void VSSBlobFunctions::Register(ExtensionLoader &loader) {
	// Utility functions
	loader.RegisterFunction(ScalarFunction("duckir_vss_encode_blob", {LogicalType::LIST(LogicalType::FLOAT)},
	                                       LogicalType::BLOB, EncodeBlobFunction));

	loader.RegisterFunction(ScalarFunction("duckir_vss_decode_blob", {LogicalType::BLOB},
	                                       LogicalType::LIST(LogicalType::FLOAT), DecodeBlobFunction));

	loader.RegisterFunction(
	    ScalarFunction("duckir_vss_blob_to_base64", {LogicalType::BLOB}, LogicalType::VARCHAR, BlobToBase64Function));

	loader.RegisterFunction(
	    ScalarFunction("duckir_vss_base64_to_blob", {LogicalType::VARCHAR}, LogicalType::BLOB, Base64ToBlobFunction));

	// Distance function overloads
	RegisterDistanceOverloads<BlobDistanceOp>(loader, "array_distance");
	RegisterDistanceOverloads<BlobDistanceOp>(loader, "<->");
	RegisterDistanceOverloads<BlobCosineDistanceOp>(loader, "array_cosine_distance");
	RegisterDistanceOverloads<BlobCosineDistanceOp>(loader, "<=>");
}

void VSSModule::RegisterBlobFunctions(ExtensionLoader &loader) {
	VSSBlobFunctions::Register(loader);
}

} // namespace duckdb
