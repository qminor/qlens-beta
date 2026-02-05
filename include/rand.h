#ifndef RAND_H
#define RAND_H
#include <cmath>

class Random
{
	private:
	unsigned long long int u, v, w;
	bool first_normal_deviate;
	double gset;
	unsigned long long int seed;
		
	public:
	Random()
	{
		first_normal_deviate = true;
		Random(10);
	}
	Random(unsigned long long int j) : v(4101842887655102017LL), w(1)
	{
		seed = j;
		u = j ^ v; int64();
		v = u; int64(); int64();
		w = v; int64(); int64();
		first_normal_deviate = true;
	}
	void set_random_seed(unsigned long long int j)
	{
		seed = j;
		v = 4101842887655102017LL;
		w = 1;
		u = j ^ v; int64();
		v = u; int64(); int64();
		w = v; int64(); int64();
		first_normal_deviate = true;
	}
	void reinitialize_random_generator()
	{
		v = 4101842887655102017LL;
		w = 1;
		u = seed ^ v; int64();
		v = u; int64(); int64();
		w = v; int64(); int64();
		first_normal_deviate = true;
	}
	void set_random_generator(Random* rand_in) { u = rand_in->u; v = rand_in->v; w = rand_in->w; }
	inline unsigned long long int int64()
	{
		u = u * 2862933555777941757LL + 7046029254386353087LL;
		v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
		w = 4294957665U*(w & 0xffffffff) + (w >> 32);
		unsigned long long int x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
		return (x + v) ^ w;
	}
	unsigned long long int get_random_seed() { return seed; }
	double RandomNumber() { return 5.42101086242752217E-20 * int64(); }
	inline unsigned int int32() { return (unsigned int)int64(); }
	double NormalDeviate()
	{
		double fac,rsq,v1,v2;

		if (first_normal_deviate) {
			do {
				v1=2.0*RandomNumber()-1.0;
				v2=2.0*RandomNumber()-1.0;
				rsq=v1*v1+v2*v2;
			} while ((rsq >= 1.0) or (rsq == 0.0));
			fac=sqrt(-2.0*log(rsq)/rsq);
			gset=v1*fac;
			first_normal_deviate = false;
			return v2*fac;
		} else {
			first_normal_deviate = true;
			return gset;
		}
	}
	/*
	double NormalDeviate()
	{
		double u, v, x, y, q;
		do
		{
			u = RandomNumber();
			v = 1.7156*(RandomNumber() - 0.5);
			x = u - 0.449871;
			y = fabs(v) + 0.386595;
			q = x*x + y*(0.19600*y-0.25472*x);
		}
		while(q > 0.27597 && (q > 0.27846 || v*v > -4.0*log(u)*u*u));
		
		return v/u;
	}
	*/

};



#endif // RAND_H
