#ifndef DELAUNEY_H
#define DELAUNEY_H
#include "lensvec.h"

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

const unsigned char OK = 0x01;
const unsigned char NOTOK = 0x00;
const unsigned char ONSIDE = 0x02;
const unsigned char DISCARD = 0x04;
const unsigned char BOT = 0x01;
const unsigned char NOTBOT = 0x02;
const unsigned char USED = 0x04;
const unsigned char NODEL = 0x08;
const unsigned char BI = 0x10;

template <typename QScalar>
struct Triangle // the final triangulation will be stored in an array of triangle structs
{
	lensvector<QScalar> vertex[3];
	lensvector<QScalar> circumcenter;
	double circumcircle_radsq;
	double area; // this will be a signed quantity since it's given by the cross product of the side vectors
	int vertex_index[3];
	int neighbor_index[3];
	Triangle() {}
	void copy_triangle(Triangle<double>* tri_in) {
		for (int i=0; i < 3; i++) {
			//std::cout << "INDEX " << i << "..." << std::endl;
			tri_in->vertex_index[i] = vertex_index[i];
			tri_in->neighbor_index[i] = neighbor_index[i];
#ifdef USE_STAN
			if constexpr (std::is_same_v<QScalar, stan::math::var>) {
				tri_in->vertex[i][0] = vertex[i][0].val();
				tri_in->vertex[i][1] = vertex[i][1].val();
			} else
#endif
			{
				tri_in->vertex[i][0] = vertex[i][0];
				tri_in->vertex[i][1] = vertex[i][1];
			}
		}
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>) {
			tri_in->circumcenter[0] = circumcenter[0].val();
			tri_in->circumcenter[1] = circumcenter[1].val();
		} else
#endif
		{
			tri_in->circumcenter[0] = circumcenter[0];
			tri_in->circumcenter[1] = circumcenter[1];
		}
		tri_in->circumcircle_radsq = circumcircle_radsq;
		tri_in->area = area;
	}
};

struct triangleBase;

struct triangleTree
{
	triangleTree *daughters[3];
	int vertices[3];
	unsigned char flag;
	
	triangleTree() : flag(BOT)
	{
		vertices[0] = -1;
		vertices[1] = -2;
		vertices[2] = -3;
		daughters[0] = daughters[1] = daughters[2] = NULL;
	}
	
	triangleTree(int *vertsin) : flag(BOT)
	{
		vertices[0] = vertsin[0];
		vertices[1] = vertsin[1];
		vertices[2] = vertsin[2];
		daughters[0] =  daughters[1] =  daughters[2] = NULL;
	}
	
	~triangleTree()
	{
		if (!bool(flag&NODEL) && !bool(flag&BOT))
			for (int i = 0; i < 3; i++)
            {
                if (daughters[i] != NULL)
                    delete daughters[i];
            }
	}
};

struct triangleBase
{
	triangleBase *sides[3];
	triangleTree *tri;
	int index;
	unsigned char sidesindices[3];
	int neighbor_indices[3];
	
	triangleBase() : tri(NULL)
	{
		sides[0] = sides[1] = sides[2] = NULL;
		neighbor_indices[0] = neighbor_indices[1] = neighbor_indices[2] = -1;
		index = -1;
	}
	
	void Shift(const int n)
	{
		if (n==2)
		{
			triangleBase *tempTri = sides[0];
			int tempIn = sidesindices[0];
			sides[0] = sides[1];
			sides[1] = sides[2];
			sides[2] = tempTri;
			sidesindices[0] = sidesindices[1];
			sidesindices[1] = sidesindices[2];
			sidesindices[2] = tempIn;
		}
		else if (n==0)
		{
			triangleBase *tempTri = sides[0];
			int tempIn = sidesindices[0];
			sides[0] = sides[2];
			sides[2] = sides[1];
			sides[1] = tempTri;
			sidesindices[0] = sidesindices[2];
			sidesindices[2] = sidesindices[1];
			sidesindices[1] = tempIn;
		}
	}
};

template <typename QScalar>
class Delaunay
{
	private:
		triangleTree *tris;
		triangleBase *botTris;
		triangleBase *botPtr;
		
	public:
		QScalar *x;
		QScalar *y;
		int nTris;
		int nvertex;

		Delaunay(QScalar *xin, QScalar *yin, const int n) : x(xin), y(yin), nvertex(n), nTris(2*n+1)
		{
			tris = new triangleTree();
			botPtr = botTris = new triangleBase[nTris];
			botPtr->tri = tris;
			tris->daughters[0] = (triangleTree *)botPtr++;
		}
		
		inline unsigned char Test(triangleTree *tri, const int pt)
		{
			double temp = 1.0;
			unsigned char result = OK;
			bool ti, tj;
			static int i1[3] = {1, 2, 0};
			static double xInfty[4] = {0.0, 0.0, -1.732, 1.732};
			static double yInfty[4] = {0.0, 1.0, -1.0, -1.0};
			
			for (int i = 0; i < 3; i++)
			{
				ti = tri->vertices[i] >= 0;
				tj = tri->vertices[i1[i]] >= 0;
				
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					if (ti&&tj)
					{
						temp = (y[pt].val()-y[tri->vertices[i]].val())*(x[tri->vertices[i1[i]]].val() - x[tri->vertices[i]].val()) - (x[pt].val()-x[tri->vertices[i]].val())*(y[tri->vertices[i1[i]]].val() - y[tri->vertices[i]].val());
					}	
					else if (ti)
					{
						temp = (y[pt].val()-y[tri->vertices[i]].val())*xInfty[-tri->vertices[i1[i]]] - (x[pt].val()-x[tri->vertices[i]].val())*yInfty[-tri->vertices[i1[i]]];
					}
					else if (tj)
					{
						temp = (x[pt].val()-x[tri->vertices[i1[i]]].val())*yInfty[-tri->vertices[i]] - (y[pt].val()-y[tri->vertices[i1[i]]].val())*xInfty[-tri->vertices[i]];
					}
				} else
#endif
				{
					if (ti&&tj)
					{
						temp = (y[pt]-y[tri->vertices[i]])*(x[tri->vertices[i1[i]]] - x[tri->vertices[i]])- (x[pt]-x[tri->vertices[i]])*(y[tri->vertices[i1[i]]] - y[tri->vertices[i]]);
					}	
					else if (ti)
					{
						temp = (y[pt]-y[tri->vertices[i]])*xInfty[-tri->vertices[i1[i]]]- (x[pt]-x[tri->vertices[i]])*yInfty[-tri->vertices[i1[i]]];
					}
					else if (tj)
					{
						temp = (x[pt]-x[tri->vertices[i1[i]]])*yInfty[-tri->vertices[i]]- (y[pt]-y[tri->vertices[i1[i]]])*xInfty[-tri->vertices[i]];
					}
				}
				
				if (temp < 0)
					return NOTOK;
				else if (temp == 0)
				{
					if (result&ONSIDE)
						return DISCARD;
					else
					{
						result |= (ONSIDE | (0x10 << ((i+2)%3)));
					}
						
				}	
			}
			return result;
		}
		
		inline bool Change(triangleTree *tri, const int in)
		{
			bool vert1 = tri->vertices[1] >= 0;
			bool vert2 = tri->vertices[2] >= 0;
			
			if (in < 0)
				return false;
			else if (vert1&&vert2)
			{
				double a0, a1, c0, c1, det, asq, csq, ctr0, ctr1, rad2, d2;
		
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					a0 = x[tri->vertices[0]].val()-x[tri->vertices[1]].val();
					a1 = y[tri->vertices[0]].val()-y[tri->vertices[1]].val();
					c0 = x[tri->vertices[2]].val()-x[tri->vertices[1]].val();
					c1 = y[tri->vertices[2]].val()-y[tri->vertices[1]].val();
				} else
#endif
				{
					a0 = x[tri->vertices[0]]-x[tri->vertices[1]];
					a1 = y[tri->vertices[0]]-y[tri->vertices[1]];
					c0 = x[tri->vertices[2]]-x[tri->vertices[1]];
					c1 = y[tri->vertices[2]]-y[tri->vertices[1]];
				}
				det = 0.5/(a0*c1-c0*a1);
				asq = a0*a0 + a1*a1;
				csq = c0*c0 + c1*c1;
				ctr0 = det*(asq*c1 - csq*a1);
				ctr1 = det*(csq*a0 - asq*c0);
				rad2 = ctr0*ctr0 + ctr1*ctr1;
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					d2 = (ctr0 + x[tri->vertices[1]].val() - x[in].val())*(ctr0 + x[tri->vertices[1]].val() - x[in].val())+(ctr1 + y[tri->vertices[1]].val() - y[in].val())*(ctr1 + y[tri->vertices[1]].val() - y[in].val());
				} else
#endif
				{
					d2 = (ctr0 + x[tri->vertices[1]] - x[in])*(ctr0 + x[tri->vertices[1]] - x[in])+(ctr1 + y[tri->vertices[1]] - y[in])*(ctr1 + y[tri->vertices[1]] - y[in]);
				}
				return (rad2 > d2);
			}
			else if (vert1)
			{
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					return (x[tri->vertices[1]].val() - x[tri->vertices[0]].val())*(y[in].val() - y[tri->vertices[0]].val())-(x[in].val() - x[tri->vertices[0]].val())*(y[tri->vertices[1]].val() - y[tri->vertices[0]].val()) > 0;
				} else
#endif
				{
					return (x[tri->vertices[1]] - x[tri->vertices[0]])*(y[in] - y[tri->vertices[0]])-(x[in] - x[tri->vertices[0]])*(y[tri->vertices[1]] - y[tri->vertices[0]]) > 0;
				}
			}
			else if (vert2)
			{
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					return ((x[in].val() - x[tri->vertices[0]].val())*(y[tri->vertices[2]].val() - y[tri->vertices[0]].val())-(x[tri->vertices[2]].val() - x[tri->vertices[0]].val())*(y[in].val() - y[tri->vertices[0]].val())) > 0;
				} else
#endif
				{
					return ((x[in] - x[tri->vertices[0]])*(y[tri->vertices[2]] - y[tri->vertices[0]])-(x[tri->vertices[2]] - x[tri->vertices[0]])*(y[in] - y[tri->vertices[0]])) > 0;
				}
			}
			else 
				return true;
		}
		
		inline triangleTree *SwitchSides(triangleTree *src, triangleTree *des, const int desIndex)
		{
			int pts[3];
			triangleBase *srcBase = (triangleBase *)src->daughters[0];
			triangleBase *desBase = (triangleBase *)des->daughters[0];
			des->flag = NOTBOT|BI;
			src->flag = NOTBOT|NODEL|BI;
			pts[0] = src->vertices[0];
			pts[1] = src->vertices[1];
			pts[2] = des->vertices[desIndex];
			des->daughters[0] = src->daughters[0] = new triangleTree(pts);
			pts[0] = src->vertices[0];
			pts[1] = des->vertices[desIndex];
			pts[2] = src->vertices[2];
			des->daughters[1] = src->daughters[1] = new triangleTree(pts);

			srcBase->tri = src->daughters[0];
			src->daughters[0]->daughters[0] = (triangleTree *)srcBase;
			desBase->tri = src->daughters[1];
			src->daughters[1]->daughters[0] = (triangleTree *)desBase;
			desBase->Shift(desIndex);
			
			if (desBase->sides[2] != NULL)
			{
				srcBase->sidesindices[0] = desBase->sidesindices[2];
				srcBase->sides[0] = desBase->sides[2];
				desBase->sides[2]->sides[desBase->sidesindices[2]] = srcBase;
				desBase->sides[2]->sidesindices[desBase->sidesindices[2]] = 0;
			}
			
			desBase->sides[0]->sidesindices[desBase->sidesindices[0]] = 0;
			
			desBase->sidesindices[1] = srcBase->sidesindices[1];
			desBase->sides[1] = srcBase->sides[1];
			srcBase->sides[1]->sides[srcBase->sidesindices[1]] = desBase;
			srcBase->sides[1]->sidesindices[srcBase->sidesindices[1]] = 1;
			
			srcBase->sidesindices[1] = 2;
			srcBase->sides[1] = desBase;
			desBase->sidesindices[2]  = 1;
			desBase->sides[2] = srcBase;
			
			return src->daughters[0];
		}
		
		inline void InputPoint (const int pt)
		{
			int i, j;
			unsigned char test = NOTOK;
			triangleTree *currentTri = tris, *startTri;
			triangleBase * currentDBase[3], *currentBase;
			
			while (!(BOT&currentTri->flag))
			{
				j = (currentTri->flag&BI) ? 2: 3;
				for (i = 0; i < j; i++)
				{
					if (OK&(test = Test(currentTri->daughters[i], pt)))
					{
						currentTri = currentTri->daughters[i];
						break;
					}
					else if (DISCARD&test)
					{
						nTris -= 2;
						return;
					}
				}
			}
			
			int pts[3];
			currentTri->flag = NOTBOT;
			static int i1[3] = {1, 2, 0};
			static int i2[3] = {2, 0, 1};
			
			currentDBase[0] = (triangleBase *)currentTri->daughters[0];
			
			for (i = 0; i < 3; i++)
			{
				pts[0] = pt, pts[1] = currentTri->vertices[i1[i]], pts[2] = currentTri->vertices[i2[i]];
				currentTri->daughters[i] = new triangleTree(pts);
			}
			
			currentDBase[0]->tri = currentTri->daughters[0];
			currentTri->daughters[0]->daughters[0] = (triangleTree *)currentDBase[0];
			botPtr->tri = currentTri->daughters[1];
			currentTri->daughters[1]->daughters[0] = (triangleTree *)(currentDBase[1] = botPtr++);
			botPtr->tri = currentTri->daughters[2];
			currentTri->daughters[2]->daughters[0] = (triangleTree *)(currentDBase[2] = botPtr++);
			
			for (i = 1; i < 3; i++)
			{
				currentDBase[i]->sidesindices[1] = 2;
				currentDBase[i]->sides[1] = currentDBase[i1[i]];
				currentDBase[i]->sidesindices[2] = 1;
				currentDBase[i]->sides[2] = currentDBase[i2[i]];
				if ((currentDBase[0]->sides[i] != NULL))
				{
					currentDBase[i]->sides[0] = currentDBase[0]->sides[i];
					currentDBase[i]->sidesindices[0] = currentDBase[0]->sidesindices[i];
					currentDBase[0]->sides[i]->sides[currentDBase[0]->sidesindices[i]] = currentDBase[i];
					currentDBase[0]->sides[i]->sidesindices[currentDBase[0]->sidesindices[i]] = 0;
				}
			}
			
			currentDBase[0]->sidesindices[1] = 2;
			currentDBase[0]->sides[1] = currentDBase[1];
			currentDBase[0]->sidesindices[2] = 1;
			currentDBase[0]->sides[2] = currentDBase[2];
			
			if (test&ONSIDE)
			{
				j=0;
				while (!((test >> j)&0x10))
					j++;
				triangleTree *badTri = currentTri->daughters[j];
				triangleTree *saveTri = currentTri;
				currentTri = startTri = currentTri->daughters[0];
				currentBase = (triangleBase *)currentTri->daughters[0];
				while(true)
				{
					if ((currentBase->sides[0] != NULL)&&((badTri == currentTri)||Change(currentTri, currentBase->sides[0]->tri->vertices[currentBase->sidesindices[0]])))
					{
						currentTri = startTri = SwitchSides(currentTri, currentBase->sides[0]->tri, currentBase->sidesindices[0]);
						currentBase = (triangleBase *)currentTri->daughters[0];
					}
					else
					{
						break;
					}
				}
				currentTri = currentBase->sides[1]->tri;
				currentBase = (triangleBase *)currentTri->daughters[0];
				
				while(currentTri != startTri)
				{
					if ((currentBase->sides[0] != NULL)&&((badTri == currentTri)||Change(currentTri, currentBase->sides[0]->tri->vertices[currentBase->sidesindices[0]])))
						currentTri = SwitchSides(currentTri, currentBase->sides[0]->tri, currentBase->sidesindices[0]);
					else
					{
						currentTri = currentBase->sides[1]->tri;
						currentBase = (triangleBase *)currentTri->daughters[0];
					}
				}
				
				delete saveTri->daughters[j];
				saveTri->daughters[j] = saveTri->daughters[2];
				saveTri->daughters[2] = NULL;
				saveTri->flag |= BI;
			}
			else
			{
				currentTri = startTri = currentTri->daughters[0];
				currentBase = (triangleBase *)currentTri->daughters[0];
				while(true)
				{
					if ((currentBase->sides[0] != NULL)&&Change(currentTri, currentBase->sides[0]->tri->vertices[currentBase->sidesindices[0]]))
					{
						currentTri = startTri = SwitchSides(currentTri, currentBase->sides[0]->tri, currentBase->sidesindices[0]);
						currentBase = (triangleBase *)currentTri->daughters[0];
					}
					else
					{
						break;
					}
				}
				currentTri = currentBase->sides[1]->tri;
				currentBase = (triangleBase *)currentTri->daughters[0];
				
				while(currentTri != startTri)
				{
					if ((currentBase->sides[0] != NULL)&&Change(currentTri, currentBase->sides[0]->tri->vertices[currentBase->sidesindices[0]]))
						currentTri = SwitchSides(currentTri, currentBase->sides[0]->tri, currentBase->sidesindices[0]);
					else
					{
						currentTri = currentBase->sides[1]->tri;
						currentBase = (triangleBase *)currentTri->daughters[0];
					}
				}
			}
			
			return;
		}
		
		void Process()
		{
			const int N = (nTris - 1)/2;
			for (int i = 0; i < N; i++)
			{
				InputPoint(i);
			}
		}
		
		int TriNum()
		{
			int n = 0;
			botPtr = botTris;
			for  (int i = 0; i < nTris; i++, botPtr++)
			{
				if (botPtr->tri->vertices[0] >= 0 && botPtr->tri->vertices[1] >= 0 && botPtr->tri->vertices[2] >= 0) {
					botPtr->index = n++;
				}
			}
			return n;
		}
		
		void InputValues(int *output)
		{
			botPtr = botTris;
			for (int i = 0; i < nTris; i++, botPtr++)
			{
				if (botPtr->tri->vertices[0] >= 0 && botPtr->tri->vertices[1] >= 0 && botPtr->tri->vertices[2] >= 0)
				{
					*output++ = botPtr->tri->vertices[0];
					*output++ = botPtr->tri->vertices[1];
					*output++ = botPtr->tri->vertices[2];
				}
			}
		}	
		void store_triangles(Triangle<QScalar> *triangle)
		{
			// This stores the triangles in the final triangulation
			botPtr = botTris;
			lensvector<QScalar> side1, side2;
			for (int i = 0; i < nTris; i++, botPtr++)
			{
				QScalar a0, a1, c0, c1, det_inv, asq, csq, ctr0, ctr1;
				if (botPtr->tri->vertices[0] >= 0 && botPtr->tri->vertices[1] >= 0 && botPtr->tri->vertices[2] >= 0)
				{
					triangle->vertex_index[0] = botPtr->tri->vertices[0];
					triangle->vertex_index[1] = botPtr->tri->vertices[1];
					triangle->vertex_index[2] = botPtr->tri->vertices[2];
					triangle->vertex[0][0] = x[botPtr->tri->vertices[0]];
					triangle->vertex[0][1] = y[botPtr->tri->vertices[0]];
					triangle->vertex[1][0] = x[botPtr->tri->vertices[1]];
					triangle->vertex[1][1] = y[botPtr->tri->vertices[1]];
					triangle->vertex[2][0] = x[botPtr->tri->vertices[2]];
					triangle->vertex[2][1] = y[botPtr->tri->vertices[2]];
					//triangle->index = botPtr->index;
					if (botPtr->sides[0] != NULL) {
						triangle->neighbor_index[0] = botPtr->sides[0]->index; // not sure if sides is *ever* NULL, but just in case...
						//triangle->midpoint[0] = (triangle->vertex[1] + triangle->vertex[2])/2;
					} else triangle->neighbor_index[0] = -1;
					if (botPtr->sides[1] != NULL) {
						triangle->neighbor_index[1] = botPtr->sides[1]->index;
						//triangle->midpoint[1] = (triangle->vertex[0] + triangle->vertex[2])/2;
					} else triangle->neighbor_index[1] = -1;
					if (botPtr->sides[2] != NULL) {
						triangle->neighbor_index[2] = botPtr->sides[2]->index;
						//triangle->midpoint[2] = (triangle->vertex[0] + triangle->vertex[1])/2;
					} else triangle->neighbor_index[2] = -1;
					side1 = triangle->vertex[1] - triangle->vertex[0];
					side2 = triangle->vertex[2] - triangle->vertex[1];
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					triangle->area = (side1 ^ side2).val();
				} else
#endif
				{
					triangle->area = side1 ^ side2;
				}

					a0 = x[botPtr->tri->vertices[0]]-x[botPtr->tri->vertices[1]];
					a1 = y[botPtr->tri->vertices[0]]-y[botPtr->tri->vertices[1]];
					c0 = x[botPtr->tri->vertices[2]]-x[botPtr->tri->vertices[1]];
					c1 = y[botPtr->tri->vertices[2]]-y[botPtr->tri->vertices[1]];
					det_inv = 0.5/(a0*c1-c0*a1);
					asq = a0*a0 + a1*a1;
					csq = c0*c0 + c1*c1;
					ctr0 = det_inv*(asq*c1 - csq*a1);
					ctr1 = det_inv*(csq*a0 - asq*c0);
					triangle->circumcenter[0] = ctr0 + x[botPtr->tri->vertices[1]];
					triangle->circumcenter[1] = ctr1 + y[botPtr->tri->vertices[1]];
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					triangle->circumcircle_radsq = ctr0.val()*ctr0.val()+ctr1.val()*ctr1.val();
				} else
#endif
				{
					triangle->circumcircle_radsq = ctr0*ctr0+ctr1*ctr1;
				}
					triangle++;
				}
			}

		}
		~Delaunay()
		{
			delete tris;
			delete[] botTris;
		}
};

#endif
