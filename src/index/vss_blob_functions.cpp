#include "index/vss_blob_functions.hpp"
#include "index/vss_blob_codec.hpp"
#include "index/vss_module.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
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

// Reads the text of a query vector as '[…]'::FLOAT[] does, and returns its float32 values as a BLOB.
static void EncodeQueryFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	Vector input(LogicalType::LIST(LogicalType::FLOAT), count);
	VectorOperations::Cast(state.GetContext(), args.data[0], input, count);

	UnifiedVectorFormat input_format;
	input.ToUnifiedFormat(input_format);

	auto &child_vec = ListVector::GetChild(input);
	auto child_data = FlatVector::GetData<float>(child_vec);
	auto &child_validity = FlatVector::Validity(child_vec);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(input_format);

	for (idx_t i = 0; i < count; i++) {
		auto idx = input_format.sel->get_index(i);
		if (!input_format.validity.RowIsValid(idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto &entry = list_entries[idx];
		for (idx_t j = entry.offset; j < entry.offset + entry.length; j++) {
			if (!child_validity.RowIsValid(j)) {
				throw InvalidInputException("duckir_vss_encode_query: the query vector can not contain NULL values");
			}
		}
		auto blob_size = entry.length * sizeof(float);
		auto blob = StringVector::EmptyString(result, blob_size);
		EncodeFloatArrayToBlob(child_data + entry.offset, entry.length, data_ptr_cast(blob.GetDataWriteable()));
		blob.Finalize();
		FlatVector::GetDataMutable<string_t>(result)[i] = blob;
	}

	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
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

// Resolve types for (BLOB, ARRAY) - blob is arg 0, array is arg 1
static void BlobArrayDistanceResolveTypes(ResolveScalarFunctionTypesInput &input) {
	auto &bound_function = input.GetBoundFunction();
	auto &array_type = input.GetArgumentType(1);
	if (array_type.id() != LogicalTypeId::ARRAY) {
		throw BinderException("%s: second argument must be an ARRAY type",
		                      bound_function.GetName().GetIdentifierName());
	}
	auto array_size = ArrayType::GetSize(array_type);
	bound_function.GetArguments()[1] = LogicalType::ARRAY(LogicalType::FLOAT, array_size);
}

// Bind for (BLOB, ARRAY) - blob is arg 0, array is arg 1
static unique_ptr<FunctionData> BlobArrayDistanceBind(BindScalarFunctionInput &input) {
	return make_uniq<BlobDistanceBindData>(ArrayType::GetSize(input.GetBoundFunction().GetArguments()[1]));
}

// Resolve types for (ARRAY, BLOB) - array is arg 0, blob is arg 1
static void ArrayBlobDistanceResolveTypes(ResolveScalarFunctionTypesInput &input) {
	auto &bound_function = input.GetBoundFunction();
	auto &array_type = input.GetArgumentType(0);
	if (array_type.id() != LogicalTypeId::ARRAY) {
		throw BinderException("%s: first argument must be an ARRAY type", bound_function.GetName().GetIdentifierName());
	}
	auto array_size = ArrayType::GetSize(array_type);
	bound_function.GetArguments()[0] = LogicalType::ARRAY(LogicalType::FLOAT, array_size);
}

// Bind for (ARRAY, BLOB) - array is arg 0, blob is arg 1
static unique_ptr<FunctionData> ArrayBlobDistanceBind(BindScalarFunctionInput &input) {
	return make_uniq<BlobDistanceBindData>(ArrayType::GetSize(input.GetBoundFunction().GetArguments()[0]));
}

// Execute: (BLOB, ARRAY<FLOAT>[N]) → FLOAT
template <class OP>
static void BlobArrayDistanceExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_state = state.Cast<ExecuteFunctionState>();
	auto &bind_data = func_state.expr.Cast<BoundFunctionExpression>().BindInfo()->Cast<BlobDistanceBindData>();
	auto array_size = bind_data.array_size;

	auto count = args.size();
	auto &blob_vec = args.data[0];
	auto &array_vec = args.data[1];

	UnifiedVectorFormat blob_format;
	blob_vec.ToUnifiedFormat(blob_format);
	auto blob_data = UnifiedVectorFormat::GetData<string_t>(blob_format);

	UnifiedVectorFormat array_format;
	array_vec.ToUnifiedFormat(array_format);

	auto &array_child = ArrayVector::GetChild(array_vec);
	auto array_child_data = FlatVector::GetData<float>(array_child);

	auto res_data = FlatVector::GetDataMutable<float>(result);

	auto decoded = make_unsafe_uniq_array<float>(array_size);
	optional_idx decoded_blob_idx;

	for (idx_t i = 0; i < count; i++) {
		auto blob_idx = blob_format.sel->get_index(i);
		auto array_idx = array_format.sel->get_index(i);

		if (!blob_format.validity.RowIsValid(blob_idx) || !array_format.validity.RowIsValid(array_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		// A constant query vector is copied once per chunk.
		if (!decoded_blob_idx.IsValid() || decoded_blob_idx.GetIndex() != blob_idx) {
			auto &blob = blob_data[blob_idx];
			if (blob.GetSize() != array_size * sizeof(float)) {
				throw InvalidInputException("BLOB of %llu bytes does not hold the %llu float32 values of the array",
				                            blob.GetSize(), array_size);
			}
			DecodeBlobToFloatArray(const_data_ptr_cast(blob.GetData()), blob.GetSize(), decoded.get());
			decoded_blob_idx = blob_idx;
		}
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
	auto &bind_data = func_state.expr.Cast<BoundFunctionExpression>().BindInfo()->Cast<BlobDistanceBindData>();
	auto array_size = bind_data.array_size;

	auto count = args.size();
	auto &array_vec = args.data[0];
	auto &blob_vec = args.data[1];

	UnifiedVectorFormat array_format;
	array_vec.ToUnifiedFormat(array_format);
	auto &array_child = ArrayVector::GetChild(array_vec);
	auto array_child_data = FlatVector::GetData<float>(array_child);

	UnifiedVectorFormat blob_format;
	blob_vec.ToUnifiedFormat(blob_format);
	auto blob_data = UnifiedVectorFormat::GetData<string_t>(blob_format);

	auto res_data = FlatVector::GetDataMutable<float>(result);

	auto decoded = make_unsafe_uniq_array<float>(array_size);
	optional_idx decoded_blob_idx;

	for (idx_t i = 0; i < count; i++) {
		auto array_idx = array_format.sel->get_index(i);
		auto blob_idx = blob_format.sel->get_index(i);

		if (!array_format.validity.RowIsValid(array_idx) || !blob_format.validity.RowIsValid(blob_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		// A constant query vector is copied once per chunk.
		if (!decoded_blob_idx.IsValid() || decoded_blob_idx.GetIndex() != blob_idx) {
			auto &blob = blob_data[blob_idx];
			if (blob.GetSize() != array_size * sizeof(float)) {
				throw InvalidInputException("BLOB of %llu bytes does not hold the %llu float32 values of the array",
				                            blob.GetSize(), array_size);
			}
			DecodeBlobToFloatArray(const_data_ptr_cast(blob.GetData()), blob.GetSize(), decoded.get());
			decoded_blob_idx = blob_idx;
		}
		res_data[i] =
		    OP::template Operation<float>(array_child_data + array_idx * array_size, decoded.get(), array_size);
	}

	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

template <class OP>
static void RegisterDistanceOverloads(ExtensionLoader &loader, const Identifier &func_name) {
	auto array_type = LogicalType::ARRAY(LogicalType::FLOAT, optional_idx());

	// (BLOB, ARRAY<FLOAT>[N]) → FLOAT
	ScalarFunction blob_array(func_name, {LogicalType::BLOB, array_type}, LogicalType::FLOAT,
	                          BlobArrayDistanceExecute<OP>, BlobArrayDistanceBind);
	blob_array.SetResolveTypesCallback(BlobArrayDistanceResolveTypes);
	blob_array.SetFallible();
	loader.AddFunctionOverload(blob_array);

	// (ARRAY<FLOAT>[N], BLOB) → FLOAT
	ScalarFunction array_blob(func_name, {array_type, LogicalType::BLOB}, LogicalType::FLOAT,
	                          ArrayBlobDistanceExecute<OP>, ArrayBlobDistanceBind);
	array_blob.SetResolveTypesCallback(ArrayBlobDistanceResolveTypes);
	array_blob.SetFallible();
	loader.AddFunctionOverload(array_blob);
}

//------------------------------------------------------------------------------
// Registration
//------------------------------------------------------------------------------

void VSSBlobFunctions::Register(ExtensionLoader &loader) {
	// Utility functions
	ScalarFunction encode_query("duckir_vss_encode_query", {LogicalType::VARCHAR}, LogicalType::BLOB,
	                            EncodeQueryFunction);
	encode_query.SetFallible();
	loader.RegisterFunction(encode_query);

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
