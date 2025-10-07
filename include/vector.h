// VECTOR.H: Vector template

#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include "errors.h"

template <class T>
class Vector
{
	int nn;

	public:
	T *v;
	Vector() : nn(0), v(0) {}
	explicit Vector(const int &n) : v(0) { input(n); }
	Vector(T *v_in, const int &n) : v(0) { input(v_in, n); }
	Vector(const Vector&); // copy-constructor
	void input(const Vector&); // copy-constructor
	inline Vector& operator = (const Vector&);
	inline Vector& operator = (const T&);
	void input(const int&);
	void input_zero(const int&);
	void resize(const int &n);
	void input(T*, const int&);
	T* array(void) { return v; }
	T* array_offset(const int offset) { return (v+offset); }
	T* array(void) const { return v; }
	T* end(void) { return v+nn; }
	inline T norm(void);
	int size(void) const { return nn; }
	void print(void);
~Vector();

	T& operator [] (const int n) { return v[n]; }
	T& operator [] (const int n) const { return v[n]; }
		// Note: the trailing 'const' above overloads [] so that it can be
		// used in functions where 'const Vectors' are taken as arguments

	inline T operator * (const Vector&);
	inline Vector operator * (const T);
	inline Vector operator + (const Vector&);
	inline Vector operator - (const Vector&);
	inline Vector& operator += (const Vector&);
	inline Vector& operator -= (const Vector&);
	inline Vector operator - (void);
};

typedef Vector<float> fvector;
typedef Vector<double> dvector;
typedef Vector<long double> lvector;
typedef Vector<int> ivector;
typedef Vector<long long int> livector;
typedef Vector<bool> boolvector;

template <class T>
void Vector<T>::input(const int &n)
{
	if (v != NULL)
		delete[] v;
	nn = n;
	v = new T[nn];
	return;
}

template <class T>
void Vector<T>::input_zero(const int &n)
{
	if (v != NULL)
		delete[] v;
	nn = n;
	v = new T[nn];
	for (int i=0; i < nn; i++) v[i] = 0;
	return;
}

template <class T>
void Vector<T>::resize(const int &n)
{
	// if resizing to something larger, keeps all previous entries; it resizing to something
	// smaller, keeps entries up to desired index
	T *new_v = new T[n];
	if (v != NULL) {
		for (int i=0; i < ((n < nn) ? n : nn); i++) { new_v[i] = v[i]; }
		delete[] v;
	}
	nn = n;
	v = new_v;
	return;
}

template <class T>
void Vector<T>::input(T *v_in, const int &n)
{
	if (v != NULL)
		delete[] v;
	nn = n;
	v = new T[nn];
	for (int i=0; i < nn; i++)
		v[i] = v_in[i];
	return;
}

template <class T>
Vector<T>::Vector(const Vector<T>& w)
{
	nn = w.nn;
	v = new T[nn];
	for (int i=0; i < nn; i++)
		v[i] = w.v[i];
	return;
}

template <class T>
void Vector<T>::input(const Vector<T>& w)
{
	nn = w.nn;
	v = new T[nn];
	for (int i=0; i < nn; i++)
		v[i] = w.v[i];
	return;
}

template <class T>
inline Vector<T>& Vector<T>::operator= (const Vector<T>& w)
{
	if ((nn != w.nn) or (v==NULL)) {
		nn = w.nn;
		if (v != NULL)
			delete[] v;
		v = new T[nn];
	}
	for (int i=0; i < nn; i++)
		v[i] = w.v[i];
	return *this;
}

template <class T>
inline Vector<T>& Vector<T>::operator= (const T& num)
{
	for (int i=0; i < nn; i++)
		v[i] = num;
	return *this;
}

template <class T>
Vector<T>::~Vector()
{
	if (v != NULL)
		delete[] v;
	return;
}

template <class T>
void Vector<T>::print(void)
{
	for (int i=0; i < nn-1; i++)
		std::cout << v[i] << "  ";
	std::cout << v[nn-1] << std::endl;
	return;
}

template <class T>
inline T Vector<T>::operator* (const Vector& b)
{
	if (nn != b.nn) die("Vectors must have same size for dot product");
	T ans = 0;
	for (int i=0; i < nn; i++)
		ans += v[i] * b.v[i];
	return ans;
}

template <class T>
inline Vector<T> Vector<T>::operator+ (const Vector& b)
{
	if (nn != b.nn) die("Vectors must have same size for dot product");
	Vector ans(nn);
	for (int i=0; i < nn; i++)
		ans[i] = v[i] + b.v[i];
	return ans;
}

template <class T>
inline Vector<T>& Vector<T>::operator+= (const Vector& b)
{
	if (nn != b.nn) die("Vectors must have same size for dot product");
	for (int i=0; i < nn; i++)
		v[i] += b.v[i];
	return *this;
}

template <class T>
inline Vector<T> Vector<T>::operator- (const Vector& b)
{
	if (nn != b.nn) die("Vectors must have same size for dot product");
	Vector ans(nn);
	for (int i=0; i < nn; i++)
		ans[i] = v[i] - b.v[i];
	return ans;
}

template <class T>
inline Vector<T>& Vector<T>::operator-= (const Vector& b)
{
	if (nn != b.nn) die("Vectors must have same size for dot product");
	for (int i=0; i < nn; i++)
		v[i] -= b.v[i];
	return *this;
}

template <class T>
inline Vector<T> Vector<T>::operator - (void)
{
	Vector<T> ans(nn);
	for (int i=0; i < nn; i++)
		ans[i] = -v[i];
	return ans;
}

template <class T>
inline Vector<T> Vector<T>::operator * (const T num)
{
	Vector<T> b(nn);
	for (int i=0; i < nn; i++)
		b.v[i] = num * v[i];
	return b;
}

template <class T>
inline T Vector<T>::norm(void)
{
	T ans = 0;
	for (int i=0; i < nn; i++)
		ans += v[i]*v[i];
	return sqrt(ans);
}

template <class T>
inline Vector<T> operator * (const T num, const Vector<T>& a)
{
	Vector<T> b(a.size());
	for (int i=0; i < a.size(); i++)
		b[i] = num * a[i];
	return b;
}

#endif // VECTOR_H
