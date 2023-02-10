#ifndef GAUSS_H
#define GAUSS_H

// for adaptive quadrature, see "GaussPatterson" and "ClenshawCurtis" below

class GaussianIntegral
{
     protected:
         double *weights;
         double *points;
         int numberOfPoints;
			static const double EPS;
			static const double RT4M_PI_INV;
			double Gamma(const double xx);
          
     public:
			 GaussianIntegral();
          GaussianIntegral(int);
          double output(double (GaussianIntegral::*)(double));
          ~GaussianIntegral();
          double NIntegrate(double (GaussianIntegral::*)(double), double, double);
          double NIntegrateInf(double (GaussianIntegral::*)(double));
};

class GaussLegendre : public GaussianIntegral
{
     public:
		  	 GaussLegendre() : GaussianIntegral() {}
          GaussLegendre(int);
          void SetGaussLegendre(int);
			 void SetGaussLegendre(int N, double *points_in, double *weights_in);
};

class GaussHermite : public GaussianIntegral
{
     public:
		  	 GaussHermite() : GaussianIntegral() {}
          GaussHermite(int);
          void SetGaussHermite(int);
};

class GaussPatterson
{
     protected:
         double **pat_weights;
         double *pat_points;
			double *pat_funcs;
			double *pat_funcs2;
			int *pat_orders;
         int pat_N;
			double pat_tolerance, pat_tolerance_outer;
			bool show_convergence_warning;
          
     public:
			 GaussPatterson();
			 void SetGaussPatterson(const double tol_in, const bool show_warnings);
			void set_pat_tolerance_inner(const double tol_in);
			void set_pat_tolerance_outer(const double tol_in);
			void set_pat_warnings(const bool warn) { show_convergence_warning = warn; }

          ~GaussPatterson();
          double AdaptiveQuad(double (GaussPatterson::*)(double), double, double, bool& converged, bool outer = false);
};

class ClenshawCurtis
{
     protected:
         double **cc_weights;
         double *cc_points;
			double *cc_funcs;
			double *cc_funcs2;
			int *cc_lvals;
         int cc_N, cc_nlevels;
			double cc_tolerance, cc_tolerance_outer;
			bool include_endpoints; // if set to "false", use Fejer's quadarture rule (type 2) which excludes endpoints
			bool show_convergence_warning;
          
     public:
			ClenshawCurtis();
			void SetClenshawCurtis(const int nlevels_in, const double tol_in, const bool include_endpoints = true, const bool show_warnings = true);
			void set_cc_tolerance(const double tol_in);
			void set_cc_tolerance_outer(const double tol_in);
			void set_cc_warnings(const bool warn) { show_convergence_warning = warn; }

          ~ClenshawCurtis();
          double AdaptiveQuadCC(double (ClenshawCurtis::*)(double), double, double, bool& converged, bool outer = false);
};

class GaussLaguerre : public GaussianIntegral {public: GaussLaguerre(const double, int);};
class GaussJacobi : public GaussianIntegral {public: GaussJacobi(const double, const double, int);};
class GaussChebyshev : public GaussianIntegral {public: GaussChebyshev(int);};

#endif // GAUSS_H
