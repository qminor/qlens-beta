#ifndef RAND_H
#define RAND_H
#include <cmath>

/*
class Random
{
	private:
	int seed;
	bool first_normal_deviate;
	double gset;

	public:
	Random() { seed = -1; first_normal_deviate = true; }
	Random(const int& seed_in) { seed = seed_in; first_normal_deviate = true; }
	void set_random_seed(const int& seed_in) { seed = seed_in; first_normal_deviate = true; }
	double get_random_seed() { return seed; }

	double RandomNumber1();
	double RandomNumber2();
	double NormalDeviate();
};
*/

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
		inline unsigned long long int int64()
		{
			u = u * 2862933555777941757LL + 7046029254386353087LL;
			v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
			w = 4294957665U*(w & 0xffffffff) + (w >> 32);
			unsigned long long int x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
			return (x + v) ^ w;
		}
		double get_random_seed() { return seed; }
		double RandomNumber1(){return 5.42101086242752217E-20 * int64();}
		double RandomNumber2(){return RandomNumber1();}
		inline unsigned int int32(){return (unsigned int)int64();}
		inline double NormalDeviate()
		{
			double fac,rsq,v1,v2;

			if (first_normal_deviate) {
				do {
					v1=2.0*RandomNumber2()-1.0;
					v2=2.0*RandomNumber2()-1.0;
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
};



#endif // RAND_H
