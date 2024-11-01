#include "isl/ctx.h"
#include "isl/isl-noexceptions.h"
#include "isl/map.h"
#include "isl/schedule_node.h"
#include "isl/schedule_type.h"
#include <iostream>
#include <vector>

#include "cpp_functions.h"

isl_union_map *
isl_ast_generate_array_expansion_indexing(isl_schedule_node *_node,
										  isl_union_map *_extra_umap) {

	isl::schedule_node_band node =
		isl::manage_copy(_node).as<isl::schedule_node_band>();
	isl::union_map extra_umap = isl::manage(_extra_umap);
	isl_ctx *ctx = node.ctx().get();

	auto r = node.n_member();
	if (r.is_error())
		return nullptr;
	int n_members = (unsigned)r;

	r = node.get_schedule_depth();
	if (r.is_error())
		return nullptr;
	int depth = (unsigned)r;

	// TODO
	int n_expanded_dims = depth + n_members;

	isl::union_map full_array_umap;

	for (int member = 0; member < n_members; member++) {

		isl_id_to_id *array_expansion =
			isl_schedule_node_band_member_get_array_expansion(node.get(),
															  member);

		isl::union_map member_array_umap;

		auto lambda1 = [&](isl_id *id, isl_id *target_id) -> isl_stat {
			isl::id array_id = isl::manage(id);
			int factor = (int)(uintptr_t)isl::manage_copy(target_id).get_user();
			// TODO offset
			int offset = factor - 1;
			// assert(offset

			isl::union_map umap;
			extra_umap.foreach_map([&](isl::map map) {
				auto r = map.dim(isl::dim::param);
				if (r.is_error())
					return isl::stat::error();
				int n_param = (unsigned)r;
				r = map.get_space().dim(isl::dim::out);
				if (r.is_error())
					return isl::stat::error();
				int n_stmt_dims = (unsigned)r;
				n_stmt_dims = 0;
				auto stmt_id = map.get_range_tuple_id();
				isl::aff aff;
				if (factor == 1) {
					aff = isl::aff::zero_on_domain(map.domain().space());
					isl_assert(ctx, offset == 0, abort());
				} else {
					aff = isl::aff::var_on_domain(
						isl::manage(isl_local_space_from_space(
							map.domain().space().release())),
						isl::dim::set, member);
				}

				if (offset != 0)
					aff = aff.add_constant_si(offset);
				if (factor != 1)
					aff = aff.mod(factor);

				auto as_map = aff.as_map();
				as_map = as_map.add_dims(isl::dim::out,
										 n_expanded_dims - 1 - depth - member);
				as_map = isl::manage(isl_map_insert_dims(
					as_map.release(), isl_dim_out, 0, depth + member));

				as_map = as_map.set_range_tuple(array_id);

				isl::space space = as_map.get_space();
				space = space.drop_dims(isl::dim::in, 0, n_members);
				space = space.range().map_from_domain_and_range(space.domain());
				space = space.add_dims(isl::dim::out, n_stmt_dims);
				space = space.set_tuple_id(isl::dim::out, stmt_id);
				auto tag = isl::multi_aff::domain_map(space);
				as_map = as_map.preimage_range(tag);

				// auto as_umap = as_map.set_range_tuple(id).to_union_map();
				auto as_umap = as_map.to_union_map();
				if (umap.is_null())
					umap = as_umap;
				else
					umap = umap.unite(as_umap);

				return isl::stat::ok();
			});

			if (member_array_umap.is_null())
				member_array_umap = umap;
			else
				member_array_umap = member_array_umap.unite(umap);

			return isl_stat_ok;
		};
		struct Data {
			decltype(lambda1) &fn;
		} data{lambda1};
		auto lambda2 = [](isl_id *id, isl_id *target_id,
						  void *user) -> isl_stat {
			return static_cast<struct Data *>(user)->fn(id, target_id);
		};
		if (isl_id_to_id_foreach(array_expansion, lambda2, &data) < 0)
			return nullptr;

		if (full_array_umap.is_null())
			full_array_umap = member_array_umap;
		else
			// full_array_umap = full_array_umap.intersect(member_array_umap);
			full_array_umap = full_array_umap.intersect(member_array_umap);
	}

	extra_umap = extra_umap.unite(full_array_umap);

	return extra_umap.release();
}
