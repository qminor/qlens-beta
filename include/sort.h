
#ifndef SORT_H
#define SORT_H

#include "vector.h"

class Sort
{
	public:
	Sort() {}
	template <typename QScalar>
	void sort(const int, QScalar[]);
	template <typename QScalar>
	void sort(const int, QScalar[], QScalar[]);
	template <typename QScalar>
	void sort(const unsigned long, QScalar[], QScalar[]);
	template <typename QScalar>
	void sort(const int, int[], int[]);
	template <typename QScalar>
	void sort(const int n, int arr[], QScalar brr[]);
	template <typename QScalar>
	void sort(const int n, QScalar arr[], int brr[]);
	template <typename QScalar>
	void sort(const int n, const dvector& a, Vector<QScalar>& b) { sort(n,a.array(),b.array()); }
	template <typename QScalar>
	void sort(const int n, QScalar arr[], int brr[], int crr[]);
	template <typename QScalar>
	void sort(const int n, QScalar arr[], QScalar brr[], QScalar crr[]); // This is why you should use templates. Ugh
	template <typename QScalar>
	void sort(const int n, QScalar arr[], QScalar brr[], QScalar crr[], int drr[]); // This is why you should use templates. Ugh
	template <typename QScalar>
	void sort(const int n, QScalar arr[], QScalar brr[], QScalar crr[], QScalar drr[], QScalar err[]); // This is why you should use templates. Ugh
};

#endif // SORT_H
