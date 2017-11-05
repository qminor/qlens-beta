#ifndef RAND_H
#define RAND_H

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

#endif // RAND_H
