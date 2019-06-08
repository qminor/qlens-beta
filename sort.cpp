#include "sort.h"
#include "errors.h"

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;

void Sort::sort(const int n, double arr[])
{
	const int M = 7;
	const int nstack = 50;

	int i, j, k;
	int ir = n - 1, l = 0;
	int jstack = -1;
	double a, temp;

	int *istack = new int[nstack];
	for (;;) {
		if (ir-l < M) {
			for (j=l+1; j <= ir; j++) {
				a = arr[j];
				for (i = j-1; i >= l; i--) {
					if (arr[i] <= a) break;
					arr[i+1] = arr[i];
				}
				arr[i+1] = a;
			}
			if (jstack < 0) {
				delete[] istack;
				return;
			}
			ir = istack[jstack--];
			l = istack[jstack--];
		} else {
			k = (l+ir) >> 1;
			SWAP(arr[k],arr[l+1])
			if (arr[l] > arr[ir]) {
				SWAP(arr[l],arr[ir])
			}
			if (arr[l+1] > arr[ir]) {
				SWAP(arr[l+1],arr[ir])
			}
			if (arr[l] > arr[l+1]) {
				SWAP(arr[l],arr[l+1])
			}
			i = l + 1;
			j = ir;
			a = arr[l+1];
			for (;;) {
				do i++; while (arr[i] < a);
				do j--; while (arr[j] > a);
				if (j < i) break;
				SWAP(arr[i],arr[j])
			}
			arr[l+1] = arr[j];
			arr[j] = a;
			jstack += 2;
			if (jstack >= nstack) die("nstack too small in routine sort");
			if (ir-i+1 >= j-l) {
				istack[jstack] = ir;
				istack[jstack-1] = i;
				ir = j - 1;
			} else {
				istack[jstack] = j - 1;
				istack[jstack-1] = l;
				l = i;
			}
		}
	}
}

void Sort::sort(const int n, double arr[], double brr[], double crr[], double drr[], double err[]) // This is why you should use templates. Ugh
{
	const int M = 7;
	const int nstack = 50;

	int i, j, k;
	int ir = n - 1, l = 0;
	int jstack = -1;
	double a, b, c, d, e, temp;

	int *istack = new int[nstack];
	for (;;) {
		if (ir-l < M) {
			for (j=l+1; j <= ir; j++) {
				a = arr[j];
				b = brr[j];
				c = crr[j];
				d = drr[j];
				e = err[j];
				for (i = j-1; i >= l; i--) {
					if (arr[i] <= a) break;
					arr[i+1] = arr[i];
					brr[i+1] = brr[i];
					crr[i+1] = crr[i];
					drr[i+1] = drr[i];
					err[i+1] = err[i];
				}
				arr[i+1] = a;
				brr[i+1] = b;
				crr[i+1] = c;
				drr[i+1] = d;
				err[i+1] = e;
			}
			if (jstack < 0) {
				delete[] istack;
				return;
			}
			ir = istack[jstack--];
			l = istack[jstack--];
		} else {
			k = (l+ir) >> 1;
			SWAP(arr[k],arr[l+1])
			SWAP(brr[k],brr[l+1])
			SWAP(crr[k],crr[l+1])
			SWAP(drr[k],drr[l+1])
			SWAP(err[k],err[l+1])
			if (arr[l] > arr[ir]) {
				SWAP(arr[l],arr[ir])
				SWAP(brr[l],brr[ir])
				SWAP(crr[l],crr[ir])
				SWAP(drr[l],drr[ir])
				SWAP(err[l],err[ir])
			}
			if (arr[l+1] > arr[ir]) {
				SWAP(arr[l+1],arr[ir])
				SWAP(brr[l+1],brr[ir])
				SWAP(crr[l+1],crr[ir])
				SWAP(drr[l+1],drr[ir])
				SWAP(err[l+1],err[ir])
			}
			if (arr[l] > arr[l+1]) {
				SWAP(arr[l],arr[l+1])
				SWAP(brr[l],brr[l+1])
				SWAP(crr[l],crr[l+1])
				SWAP(drr[l],drr[l+1])
				SWAP(err[l],err[l+1])
			}
			i = l + 1;
			j = ir;
			a = arr[l+1];
			b = brr[l+1];
			c = crr[l+1];
			d = drr[l+1];
			e = err[l+1];
			for (;;) {
				do i++; while (arr[i] < a);
				do j--; while (arr[j] > a);
				if (j < i) break;
				SWAP(arr[i],arr[j])
				SWAP(brr[i],brr[j])
				SWAP(crr[i],crr[j])
				SWAP(drr[i],drr[j])
				SWAP(err[i],err[j])
			}
			arr[l+1] = arr[j];
			arr[j] = a;
			brr[l+1] = brr[j];
			brr[j] = b;
			crr[l+1] = crr[j];
			crr[j] = c;
			drr[l+1] = drr[j];
			drr[j] = d;
			err[l+1] = err[j];
			err[j] = e;
			jstack += 2;
			if (jstack >= nstack) die("nstack too small in routine sort2");
			if (ir-i+1 >= j-l) {
				istack[jstack] = ir;
				istack[jstack-1] = i;
				ir = j - 1;
			} else {
				istack[jstack] = j - 1;
				istack[jstack-1] = l;
				l = i;
			}
		}
	}
}



void Sort::sort(const int n, double arr[], double brr[])
{
	const int M = 7;
	const int nstack = 50;

	int i, j, k;
	int ir = n - 1, l = 0;
	int jstack = -1;
	double a, b, temp;

	int *istack = new int[nstack];
	for (;;) {
		if (ir-l < M) {
			for (j=l+1; j <= ir; j++) {
				a = arr[j];
				b = brr[j];
				for (i = j-1; i >= l; i--) {
					if (arr[i] <= a) break;
					arr[i+1] = arr[i];
					brr[i+1] = brr[i];
				}
				arr[i+1] = a;
				brr[i+1] = b;
			}
			if (jstack < 0) {
				delete[] istack;
				return;
			}
			ir = istack[jstack--];
			l = istack[jstack--];
		} else {
			k = (l+ir) >> 1;
			SWAP(arr[k],arr[l+1])
			SWAP(brr[k],brr[l+1])
			if (arr[l] > arr[ir]) {
				SWAP(arr[l],arr[ir])
				SWAP(brr[l],brr[ir])
			}
			if (arr[l+1] > arr[ir]) {
				SWAP(arr[l+1],arr[ir])
				SWAP(brr[l+1],brr[ir])
			}
			if (arr[l] > arr[l+1]) {
				SWAP(arr[l],arr[l+1])
				SWAP(brr[l],brr[l+1])
			}
			i = l + 1;
			j = ir;
			a = arr[l+1];
			b = brr[l+1];
			for (;;) {
				do i++; while (arr[i] < a);
				do j--; while (arr[j] > a);
				if (j < i) break;
				SWAP(arr[i],arr[j])
				SWAP(brr[i],brr[j])
			}
			arr[l+1] = arr[j];
			arr[j] = a;
			brr[l+1] = brr[j];
			brr[j] = b;
			jstack += 2;
			if (jstack >= nstack) die("nstack too small in routine sort2");
			if (ir-i+1 >= j-l) {
				istack[jstack] = ir;
				istack[jstack-1] = i;
				ir = j - 1;
			} else {
				istack[jstack] = j - 1;
				istack[jstack-1] = l;
				l = i;
			}
		}
	}
}

void Sort::sort(const int n, int arr[], int brr[])
{
	const int M = 7;
	const int nstack = 50;

	int i, j, k;
	int ir = n - 1, l = 0;
	int jstack = -1;
	int a, b, temp;

	int *istack = new int[nstack];
	for (;;) {
		if (ir-l < M) {
			for (j=l+1; j <= ir; j++) {
				a = arr[j];
				b = brr[j];
				for (i = j-1; i >= l; i--) {
					if (arr[i] <= a) break;
					arr[i+1] = arr[i];
					brr[i+1] = brr[i];
				}
				arr[i+1] = a;
				brr[i+1] = b;
			}
			if (jstack < 0) {
				delete[] istack;
				return;
			}
			ir = istack[jstack--];
			l = istack[jstack--];
		} else {
			k = (l+ir) >> 1;
			SWAP(arr[k],arr[l+1])
			SWAP(brr[k],brr[l+1])
			if (arr[l] > arr[ir]) {
				SWAP(arr[l],arr[ir])
				SWAP(brr[l],brr[ir])
			}
			if (arr[l+1] > arr[ir]) {
				SWAP(arr[l+1],arr[ir])
				SWAP(brr[l+1],brr[ir])
			}
			if (arr[l] > arr[l+1]) {
				SWAP(arr[l],arr[l+1])
				SWAP(brr[l],brr[l+1])
			}
			i = l + 1;
			j = ir;
			a = arr[l+1];
			b = brr[l+1];
			for (;;) {
				do i++; while (arr[i] < a);
				do j--; while (arr[j] > a);
				if (j < i) break;
				SWAP(arr[i],arr[j])
				SWAP(brr[i],brr[j])
			}
			arr[l+1] = arr[j];
			arr[j] = a;
			brr[l+1] = brr[j];
			brr[j] = b;
			jstack += 2;
			if (jstack >= nstack) die("nstack too small in routine sort2");
			if (ir-i+1 >= j-l) {
				istack[jstack] = ir;
				istack[jstack-1] = i;
				ir = j - 1;
			} else {
				istack[jstack] = j - 1;
				istack[jstack-1] = l;
				l = i;
			}
		}
	}
}

#define SWAPI(a,b) tempi=(a);(a)=(b);(b)=tempi;

void Sort::sort(const int n, double arr[], int brr[], int crr[])
{
	const int M = 7;
	const int nstack = 50;

	int i, j, k;
	int ir = n - 1, l = 0;
	int jstack = -1;
	double a, temp;
	int b, c, tempi;

	int *istack = new int[nstack];
	for (;;) {
		if (ir-l < M) {
			for (j=l+1; j <= ir; j++) {
				a = arr[j];
				b = brr[j];
				c = crr[j];
				for (i = j-1; i >= l; i--) {
					if (arr[i] <= a) break;
					arr[i+1] = arr[i];
					brr[i+1] = brr[i];
					crr[i+1] = crr[i];
				}
				arr[i+1] = a;
				brr[i+1] = b;
				crr[i+1] = c;
			}
			if (jstack < 0) {
				delete[] istack;
				return;
			}
			ir = istack[jstack--];
			l = istack[jstack--];
		} else {
			k = (l+ir) >> 1;
			SWAP(arr[k],arr[l+1])
			SWAPI(brr[k],brr[l+1])
			SWAPI(crr[k],crr[l+1])
			if (arr[l] > arr[ir]) {
				SWAP(arr[l],arr[ir])
				SWAPI(brr[l],brr[ir])
				SWAPI(crr[l],crr[ir])
			}
			if (arr[l+1] > arr[ir]) {
				SWAP(arr[l+1],arr[ir])
				SWAPI(brr[l+1],brr[ir])
				SWAPI(crr[l+1],crr[ir])
			}
			if (arr[l] > arr[l+1]) {
				SWAP(arr[l],arr[l+1])
				SWAPI(brr[l],brr[l+1])
				SWAPI(crr[l],crr[l+1])
			}
			i = l + 1;
			j = ir;
			a = arr[l+1];
			b = brr[l+1];
			c = crr[l+1];
			for (;;) {
				do i++; while (arr[i] < a);
				do j--; while (arr[j] > a);
				if (j < i) break;
				SWAP(arr[i],arr[j])
				SWAPI(brr[i],brr[j])
				SWAPI(crr[i],crr[j])
			}
			arr[l+1] = arr[j];
			arr[j] = a;
			brr[l+1] = brr[j];
			brr[j] = b;
			crr[l+1] = crr[j];
			crr[j] = c;
			jstack += 2;
			if (jstack >= nstack) die("nstack too small in routine sort2");
			if (ir-i+1 >= j-l) {
				istack[jstack] = ir;
				istack[jstack-1] = i;
				ir = j - 1;
			} else {
				istack[jstack] = j - 1;
				istack[jstack-1] = l;
				l = i;
			}
		}
	}
}

void Sort::sort(const int n, int arr[], double brr[])
{
	const int M = 7;
	const int nstack = 50;

	int i, j, k;
	int ir = n - 1, l = 0;
	int jstack = -1;
	int a, tempi;
	double b, temp;

	int *istack = new int[nstack];
	for (;;) {
		if (ir-l < M) {
			for (j=l+1; j <= ir; j++) {
				a = arr[j];
				b = brr[j];
				for (i = j-1; i >= l; i--) {
					if (arr[i] <= a) break;
					arr[i+1] = arr[i];
					brr[i+1] = brr[i];
				}
				arr[i+1] = a;
				brr[i+1] = b;
			}
			if (jstack < 0) {
				delete[] istack;
				return;
			}
			ir = istack[jstack--];
			l = istack[jstack--];
		} else {
			k = (l+ir) >> 1;
			SWAPI(arr[k],arr[l+1])
			SWAP(brr[k],brr[l+1])
			if (arr[l] > arr[ir]) {
				SWAPI(arr[l],arr[ir])
				SWAP(brr[l],brr[ir])
			}
			if (arr[l+1] > arr[ir]) {
				SWAPI(arr[l+1],arr[ir])
				SWAP(brr[l+1],brr[ir])
			}
			if (arr[l] > arr[l+1]) {
				SWAPI(arr[l],arr[l+1])
				SWAP(brr[l],brr[l+1])
			}
			i = l + 1;
			j = ir;
			a = arr[l+1];
			b = brr[l+1];
			for (;;) {
				do i++; while (arr[i] < a);
				do j--; while (arr[j] > a);
				if (j < i) break;
				SWAPI(arr[i],arr[j])
				SWAP(brr[i],brr[j])
			}
			arr[l+1] = arr[j];
			arr[j] = a;
			brr[l+1] = brr[j];
			brr[j] = b;
			jstack += 2;
			if (jstack >= nstack) die("nstack too small in routine sort2");
			if (ir-i+1 >= j-l) {
				istack[jstack] = ir;
				istack[jstack-1] = i;
				ir = j - 1;
			} else {
				istack[jstack] = j - 1;
				istack[jstack-1] = l;
				l = i;
			}
		}
	}
}

#undef SWAPI
#undef SWAP
