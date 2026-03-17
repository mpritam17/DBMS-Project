#include "rtree_node.h"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

bool almostEqual(float left, float right) {
    return std::fabs(left - right) < 1e-6f;
}

void assertBoxEquals(const BoundingBox& box, const std::vector<float>& expected_lower, const std::vector<float>& expected_upper) {
    assert(box.lower_bounds.size() == expected_lower.size());
    assert(box.upper_bounds.size() == expected_upper.size());

    for (std::size_t index = 0; index < expected_lower.size(); ++index) {
        assert(almostEqual(box.lower_bounds[index], expected_lower[index]));
        assert(almostEqual(box.upper_bounds[index], expected_upper[index]));
    }
}

}

int main() {
    RTreeNodePage leaf_node(42, 4, true);
    assert(leaf_node.isLeaf());
    assert(leaf_node.getPageId() == 42);
    assert(leaf_node.getDimensions() == 4);

    BoundingBox first_box({1.0f, 2.0f, 3.0f, 4.0f}, {1.0f, 2.0f, 3.0f, 4.0f});
    BoundingBox second_box({0.0f, 1.0f, 2.0f, 3.0f}, {2.0f, 3.0f, 4.0f, 5.0f});

    assert(leaf_node.addEntry(first_box, 1001));
    assert(leaf_node.addEntry(second_box, 1002));
    leaf_node.setParentPageId(7);
    leaf_node.setNextLeafPageId(43);

    BoundingBox node_box = leaf_node.computeNodeMBR();
    assertBoxEquals(node_box, {0.0f, 1.0f, 2.0f, 3.0f}, {2.0f, 3.0f, 4.0f, 5.0f});

    RTreeNodePage round_trip(leaf_node.getRawData());
    assert(round_trip.isLeaf());
    assert(round_trip.getPageId() == 42);
    assert(round_trip.getDimensions() == 4);
    assert(round_trip.getEntryCount() == 2);
    assert(round_trip.getParentPageId() == 7);
    assert(round_trip.getNextLeafPageId() == 43);

    RTreeEntry first_entry = round_trip.getEntry(0);
    assert(first_entry.value == 1001);
    assertBoxEquals(first_entry.mbr, {1.0f, 2.0f, 3.0f, 4.0f}, {1.0f, 2.0f, 3.0f, 4.0f});

    RTreeEntry second_entry = round_trip.getEntry(1);
    assert(second_entry.value == 1002);
    assertBoxEquals(second_entry.mbr, {0.0f, 1.0f, 2.0f, 3.0f}, {2.0f, 3.0f, 4.0f, 5.0f});

    BoundingBox hot_box = BoundingBox::point(std::vector<float>(128, 1.0f));
    RTreeNodePage high_dim_node(99, 128, false);
    assert(high_dim_node.getMaxEntries() == 15);
    for (uint16_t index = 0; index < high_dim_node.getMaxEntries(); ++index) {
        assert(high_dim_node.addEntry(hot_box, index));
    }
    assert(!high_dim_node.addEntry(hot_box, 999));

    std::cout << "R-tree node serialization test passed. Max entries @128 dims: "
              << high_dim_node.getMaxEntries() << "\n";
    return 0;
}