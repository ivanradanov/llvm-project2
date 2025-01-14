#include <algorithm>
#include <assert.h>
#include <iostream>

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "isl/isl-noexceptions.h"
#include "isl/map.h"
#include "isl/set.h"
#include "isl/space.h"
#include "isl/union_map.h"

#include "ISLUtils.h"

#define DEBUG_TYPE "isl-utils"

struct VertexMaps {
  std::vector<isl::map> out, in, intra;
};

std::pair<isl::union_map, bool> path_contract(isl::union_map umap) {
  std::vector<std::pair<isl::space, VertexMaps>> spaces;
  umap.foreach_map([&](isl::map map) {
    isl::space domain_space = map.domain().get_space();
    isl::space range_space = map.range().get_space();
    if (std::find_if(spaces.begin(), spaces.end(), [&](auto a) {
          return a.first.is_equal(domain_space);
        }) == spaces.end())
      spaces.push_back({domain_space, {}});
    if (std::find_if(spaces.begin(), spaces.end(), [&](auto a) {
          return a.first.is_equal(range_space);
        }) == spaces.end())
      spaces.push_back({range_space, {}});
    return isl::stat::ok();
  });

  LLVM_DEBUG(llvm::dbgs() << "all vertices\n");
  for (auto [k, v] : spaces) {
    LLVM_DEBUG(isl_space_dump(k.get()));
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");

  umap.foreach_map([&](isl::map map) {
    isl::space domain_space = map.domain().get_space();
    isl::space range_space = map.range().get_space();
    if (domain_space.is_equal(range_space)) {
      std::find_if(spaces.begin(), spaces.end(), [&](auto a) {
        return a.first.is_equal(domain_space);
      })->second.intra.push_back(map);
    } else {
      std::find_if(spaces.begin(), spaces.end(), [&](auto a) {
        return a.first.is_equal(domain_space);
      })->second.out.push_back(map);
      std::find_if(spaces.begin(), spaces.end(), [&](auto a) {
        return a.first.is_equal(range_space);
      })->second.in.push_back(map);
    }
    return isl::stat::ok();
  });
  for (auto [k, v] : spaces) {
    LLVM_DEBUG(llvm::dbgs() << "space ");
    LLVM_DEBUG(isl_space_dump(k.get()));
    LLVM_DEBUG(llvm::dbgs() << "in\n");
    for (auto m : v.in) {
      LLVM_DEBUG(isl_map_dump(m.get()));
    }
    LLVM_DEBUG(llvm::dbgs() << "intra\n");
    for (auto m : v.intra) {
      LLVM_DEBUG(isl_map_dump(m.get()));
    }
    LLVM_DEBUG(llvm::dbgs() << "out\n");
    for (auto m : v.out) {
      LLVM_DEBUG(isl_map_dump(m.get()));
    }

    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "pls\n");
  for (auto [k, v] : spaces) {
    LLVM_DEBUG(llvm::dbgs() << "space ");
    LLVM_DEBUG(isl_space_dump(k.get()));
    for (auto m : v.intra) {
      isl_bool exact;
      LLVM_DEBUG(llvm::dbgs() << "orig ");
      LLVM_DEBUG(isl_map_dump(m.get()));
      isl::map pl =
          isl::manage(isl_map_reaching_path_lengths(m.copy(), &exact));
      LLVM_DEBUG(llvm::dbgs() << "pl ");
      LLVM_DEBUG(isl_map_dump(pl.get()));
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  isl::union_map contracted;
  isl::space contracted_space;
  bool found_space_to_remove = false;
  for (auto [k, v] : spaces) {
    LLVM_DEBUG(llvm::dbgs() << "space ");
    LLVM_DEBUG(isl_space_dump(k.get()));

    if (v.in.size() == 0 || v.out.size() == 0) {
      // In this case we need intra path contraction which we dont have
      // currently. ignore.
      continue;
    }

    isl::map closure;
    isl::map rclosure;
    // this does not work
    // isl::space joint = isl::manage(isl_space_join(k.copy(), k.copy()));
    isl::space joint;
    bool found = false;
    for (auto m : v.intra) {
      assert(!found); // Not sure what to do with more than one intra yet
      found = true;
      joint = m.space();
      LLVM_DEBUG(llvm::dbgs() << "intra ");
      LLVM_DEBUG(isl_map_dump(m.get()));
      isl::map rintra = m.reverse();
      LLVM_DEBUG(llvm::dbgs() << "rintra ");
      LLVM_DEBUG(isl_map_dump(rintra.get()));

      isl_bool exact;
      closure = isl::manage(isl_map_transitive_closure(m.copy(), &exact));
      LLVM_DEBUG(llvm::dbgs() << "intra_c " << "exact=" << exact << " ");
      LLVM_DEBUG(isl_map_dump(closure.get()));

      rclosure = isl::manage(isl_map_transitive_closure(rintra.copy(), &exact));
      LLVM_DEBUG(llvm::dbgs() << "rintra_c " << "exact=" << exact << " ");
      LLVM_DEBUG(isl_map_dump(rclosure.get()));
    }

    if (!found) {
      assert(false);
      // TODO need to construct identity and joint and continue processing.
    }

    LLVM_DEBUG(llvm::dbgs() << "space ");
    LLVM_DEBUG(isl_space_dump(k.get()));
    LLVM_DEBUG(llvm::dbgs() << "joint ");
    LLVM_DEBUG(isl_space_dump(joint.get()));
    isl::map identity = isl::map::identity(joint);

    LLVM_DEBUG(llvm::dbgs() << "identity ");
    LLVM_DEBUG(isl_map_dump(identity.get()));

    closure = closure.unite(identity);

    LLVM_DEBUG(llvm::dbgs() << "union(intra_c, identity) ");
    LLVM_DEBUG(isl_map_dump(closure.get()));

    // for (auto intra : v.intra) {
    //   LLVM_DEBUG(llvm::dbgs() << "intra ");
    //     LLVM_DEBUG(isl_map_dump(intra.get()));
    //   LLVM_DEBUG(llvm::dbgs() << "lexmin ");
    //     LLVM_DEBUG(isl_map_dump(intra.lexmin().get()));
    //   LLVM_DEBUG(llvm::dbgs() << "lexmax ");
    //     LLVM_DEBUG(isl_map_dump(intra.lexmax().get()));
    //   LLVM_DEBUG(llvm::dbgs() << "lexmin domain ");
    //     LLVM_DEBUG(isl_set_dump(intra.domain().lexmin().get()));
    //   LLVM_DEBUG(llvm::dbgs() << "lexmax range ");
    //     LLVM_DEBUG(isl_set_dump(intra.range().lexmax().get()));
    // }
    // LLVM_DEBUG(llvm::dbgs() << "\n");

    bool remove_this_space;
    bool all_intra_covered = true;

    for (auto intra : v.intra) {
      assert(found); // Not sure what to do if there is no intra yet
      LLVM_DEBUG(llvm::dbgs() << "intra ");
      LLVM_DEBUG(isl_map_dump(intra.get()));

      bool exerciser_exists = false;

      for (auto in : v.in) {
        for (auto out : v.out) {

          // TODO think about the unity with the range, is that what we want?

          isl::set in_composition = in.range().apply(closure);
          LLVM_DEBUG(llvm::dbgs() << "closure(in_range) ");
          LLVM_DEBUG(isl_set_dump(in_composition.get()));
          in_composition = in_composition.unite(in.range());
          LLVM_DEBUG(llvm::dbgs() << "closure(in_range)  + in_range ");
          LLVM_DEBUG(isl_set_dump(in_composition.get()));

          isl::map rout = out.reverse();
          LLVM_DEBUG(llvm::dbgs() << "rout ");
          LLVM_DEBUG(isl_map_dump(rout.get()));
          isl::set out_composition = rout.range().apply(rclosure);
          LLVM_DEBUG(llvm::dbgs() << "rclosure(rout_range) ");
          LLVM_DEBUG(isl_set_dump(out_composition.get()));
          out_composition = out_composition.unite(rout.range());
          LLVM_DEBUG(llvm::dbgs() << "rclosure(rout_range)  + rout_range ");
          LLVM_DEBUG(isl_set_dump(out_composition.get()));

          if (in_composition.is_equal(out_composition)) {
            exerciser_exists = true;
            LLVM_DEBUG(llvm::dbgs() << "exerciser found\n");
          }
        }
      }

      all_intra_covered = all_intra_covered && exerciser_exists;
    }
    LLVM_DEBUG(llvm::dbgs() << "safe_to_remove=" << all_intra_covered << "\n");
    remove_this_space = all_intra_covered;
    if (!remove_this_space)
      continue;
    contracted_space = k;
    found_space_to_remove = true;
    LLVM_DEBUG(llvm::dbgs() << "\n");
    for (auto in : v.in) {
      for (auto out : v.out) {
        assert(found); // Not sure what to do if there is no intra yet
        LLVM_DEBUG(llvm::dbgs() << "in ");
        LLVM_DEBUG(isl_map_dump(in.get()));
        LLVM_DEBUG(llvm::dbgs() << "out ");
        LLVM_DEBUG(isl_map_dump(out.get()));
        isl::map comp = in.apply_range(closure);
        LLVM_DEBUG(llvm::dbgs() << "in . intra_c ");
        LLVM_DEBUG(isl_map_dump(comp.get()));
        comp = comp.apply_range(out);
        LLVM_DEBUG(llvm::dbgs() << "in . intra_c . out ");
        LLVM_DEBUG(isl_map_dump(comp.get()));

        isl::union_map umap = comp.to_union_map();

        if (contracted.is_null()) {
          contracted = umap;
        } else {
          contracted = contracted.unite(umap);
        }
      }
    }
    break;
  }

  if (!found_space_to_remove)
    return {umap, false};

  umap.foreach_map([&](isl::map map) {
    isl::space domain_space = map.domain().get_space();
    isl::space range_space = map.range().get_space();
    if (domain_space.is_equal(contracted_space) ||
        range_space.is_equal(contracted_space))
      return isl::stat::ok();
    contracted = contracted.unite(map.to_union_map());
    return isl::stat::ok();
  });

  LLVM_DEBUG(llvm::dbgs() << "contracted");
  LLVM_DEBUG(isl_union_map_dump(contracted.get()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  return {contracted, found_space_to_remove};
}

// TODO TESTS!!!!!!
isl::union_map get_maximal_paths(isl::union_map umap) {
  isl::union_map contracted = umap;
  bool did_it = true;
  while (did_it)
    std::tie(contracted, did_it) = path_contract(contracted);
  return contracted;
}
