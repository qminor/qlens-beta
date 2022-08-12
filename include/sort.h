
#ifndef SORT_H
#define SORT_H

#include "vector.h"

class Sort
{
	public:
	Sort() {}
	void sort(const int, double[]);
	void sort(const int, double[], double[]);
	void sort(const unsigned long, double[], double[]);
	void sort(const int, int[], int[]);
	void sort(const int n, int arr[], double brr[]);
	void sort(const int n, double arr[], int brr[]);
	void sort(const int n, const dvector& a, Vector<double>& b) { sort(n,a.array(),b.array()); }
	void sort(const int n, double arr[], int brr[], int crr[]);
	void sort(const int n, double arr[], double brr[], double crr[], double drr[], double err[]); // This is why you should use templates. Ugh
};

#endif // SORT_H
