#include "rand.h"
#include <cmath>

double Random::RandomNumber1()
{
	const int ia = 16807;
	const int im = 2147483647;
	const int iq = 127773;
	const int ir = 2836;
	const int ntab = 32;
	const int ndiv = (1+(im-1)/ntab);
	const double am = (1.0/im);
	const double eps = 1.2e-7;
	const double rnmx = (1.0-eps);

	static int iy = 0;
	static int iv[ntab];

	if (seed <= 0 or !iy) {
		if (-(seed) < 1) seed=1;
		else seed = -(seed);
		for (int j=ntab+7; j >= 0; j--) {
			int k = (seed) / iq;
			seed = ia*(seed-k*iq) - ir*k;
			if (seed < 0) seed += im;
			if (j < ntab) iv[j] = seed;
		}
		iy = iv[0];
	}
	int k = (seed) / iq;
	seed = ia*(seed-k*iq) - ir*k;
	if (seed < 0) seed += im;
	int j = iy / ndiv;
	iy = iv[j];
	iv[j] = seed;
	double temp;
	if ((temp = am*iy) > rnmx) return rnmx;
	else return temp;
}

double Random::RandomNumber2()
{
	const int im1=2147483563, im2=2147483399;
	const int ia1=40014, ia2=40692;
	const int iq1=53668, iq2=52774;
	const int ir1=12211, ir2=3791;
	const int ntab=32;
	const int imm1=im1-1;
	const int ndiv=1+imm1/ntab;
	const double eps=3.0e-16;
	const double rnmx=1.0-eps;
	const double am=1.0/double(im1);

	int j,k;
	static int seed2=123456789, iy=0;
	static int iv[ntab];
	double ans;

	if (seed <= 0) {	 // initialize
		seed = (seed==0 ? 1 : -seed);
		seed2 = seed;
		for (j=ntab+7; j >= 0; j--) {
			k = seed / iq1;
			seed = ia1*(seed-k*iq1) - k*ir1;
			if (seed < 0) seed += im1;
			if (j < ntab) iv[j] = seed;
		}
		iy = iv[0];
	}

	k = seed / iq1;
	seed = ia1*(seed-k*iq1) - k*ir1;	// Compute seed = (ia1*seed) % im1 without overflows by Schrage's method
	if (seed < 0) seed += im1;
	k = seed2 / iq2;
	seed2 = ia2*(seed2-k*iq2) - k*ir2;	// Compute seed2 = (ia2*seed) % im2 likewise
	if (seed2 < 0) seed2 += im2;
	j = iy / ndiv;		// j will be in the range (0,ntab-1)
	// shuffle seed, combine seed and seed2 to generate output
	iy = iv[j] - seed2;
	iv[j] = seed;
	if (iy < 1) iy += imm1;
	if ((ans=am*iy) > rnmx) return rnmx;
	else return ans;
}

double Random::NormalDeviate()
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


