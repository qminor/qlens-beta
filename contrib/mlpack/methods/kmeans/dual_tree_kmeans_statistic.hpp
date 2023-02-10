/**
 * @file methods/kmeans/dual_tree_kmeans_statistic.hpp
 * @author Ryan Curtin
 *
 * Statistic for dual-tree nearest neighbor search based k-means clustering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_DTNN_STATISTIC_HPP
#define MLPACK_METHODS_KMEANS_DTNN_STATISTIC_HPP

#include <mlpack/methods/neighbor_search/neighbor_search_stat.hpp>

namespace mlpack {

class DualTreeKMeansStatistic : public NeighborSearchStat<NearestNeighborSort>
{
 public:
  DualTreeKMeansStatistic() :
      NeighborSearchStat<NearestNeighborSort>(),
      upperBound(DBL_MAX),
      lowerBound(DBL_MAX),
      owner(size_t(-1)),
      pruned(size_t(-1)),
      staticPruned(false),
      staticUpperBoundMovement(0.0),
      staticLowerBoundMovement(0.0),
      centroid(),
		totweight(-100),
      trueParent(NULL)
  {
    // Nothing to do.
  }

  template<typename TreeType>
  DualTreeKMeansStatistic(TreeType& node) :
      NeighborSearchStat<NearestNeighborSort>(),
      upperBound(DBL_MAX),
      lowerBound(DBL_MAX),
      owner(size_t(-1)),
      pruned(size_t(-1)),
      staticPruned(false),
      staticUpperBoundMovement(0.0),
      staticLowerBoundMovement(0.0),
      trueParent(node.Parent())
  {
    // Empirically calculate the centroid.
	 //std::cout << "Getting centroids..." << endl;
    centroid.zeros(node.Dataset().n_rows);
	 totweight = 0;
	 double w;
	 bool use_weights = node.HasWeights();
	 //if (!use_weights) std::cout << "DOES NOT HAVE WEIGHTS!" << endl;
	 //bool use_weights = false;
    for (size_t i = 0; i < node.NumPoints(); ++i)
    {
      // Correct handling of cover tree: don't double-count the point which
      // appears in the children.
      if (TreeTraits<TreeType>::HasSelfChildren && i == 0 &&
          node.NumChildren() > 0)
        continue;
		  if (use_weights) {
			 w = node.Weights()[node.Point(i)];
          centroid += w*node.Dataset().col(node.Point(i));
			 totweight += w;
		  } else {
          centroid += node.Dataset().col(node.Point(i));
		  }
    }
	 //totweight = node.NumDescendants();
    //double totweight_descendants = totweight;

	 if (use_weights) {
		 for (size_t i = 0; i < node.NumChildren(); ++i) {
			w = node.Child(i).Stat().TotWeight(); 
			centroid += w * node.Child(i).Stat().Centroid();
		   totweight += w;
		 }
		 centroid /= totweight;
	 } else {
		 for (size_t i = 0; i < node.NumChildren(); ++i)
			centroid += node.Child(i).NumDescendants() *
				 node.Child(i).Stat().Centroid();
		 centroid /= node.NumDescendants();
	 }
	node.Stat().UpdateTotWeight(totweight);
	 if (use_weights) {
		 if (totweight != node.Stat().TotWeight()) std::cout << "FUCK! totweight not working right (" << totweight << " vs. " << (node.Stat().TotWeight()) << ")" << endl;
	 }

    // Set the true children correctly.
    trueChildren.resize(node.NumChildren());
    for (size_t i = 0; i < node.NumChildren(); ++i)
      trueChildren[i] = &node.Child(i);
  }

  double UpperBound() const { return upperBound; }
  double& UpperBound() { return upperBound; }

  double LowerBound() const { return lowerBound; }
  double& LowerBound() { return lowerBound; }

  const arma::vec& Centroid() const { return centroid; }
  arma::vec& Centroid() { return centroid; }

  const double& TotWeight() const { return totweight; }
  double& TotWeight() { return totweight; }

  void UpdateTotWeight(const double tw) { totweight = tw; }

  size_t Owner() const { return owner; }
  size_t& Owner() { return owner; }

  size_t Pruned() const { return pruned; }
  size_t& Pruned() { return pruned; }

  bool StaticPruned() const { return staticPruned; }
  bool& StaticPruned() { return staticPruned; }

  double StaticUpperBoundMovement() const { return staticUpperBoundMovement; }
  double& StaticUpperBoundMovement() { return staticUpperBoundMovement; }

  double StaticLowerBoundMovement() const { return staticLowerBoundMovement; }
  double& StaticLowerBoundMovement() { return staticLowerBoundMovement; }

  void* TrueParent() const { return trueParent; }
  void*& TrueParent() { return trueParent; }

  void* TrueChild(const size_t i) const { return trueChildren[i]; }
  void*& TrueChild(const size_t i) { return trueChildren[i]; }

  size_t NumTrueChildren() const { return trueChildren.size(); }

 private:
  double upperBound;
  double lowerBound;
  size_t owner;
  size_t pruned;
  bool staticPruned;
  double staticUpperBoundMovement;
  double staticLowerBoundMovement;
  arma::vec centroid;
  double totweight;
  void* trueParent;
  std::vector<void*> trueChildren;
};

} // namespace mlpack

#endif
