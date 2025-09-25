#include <complex.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define pi 3.14159265358979323846264338327

// contains stuff related to output
struct Output {
	char name[128];		// run name
	int T_print;		// interval for printing output
	int T_write;		// interval for saving state
};

// contains stuff related to data arrays
struct Arrays {
	int W;				// system width
	int H;				// system height
	ptrdiff_t lH;		// local system height
	ptrdiff_t lW;		// local system width
	ptrdiff_t lh0;		// local vertical start index
	ptrdiff_t lw0;		// local horizontal start index
	
	double u_C_min;		// minimum of smoothed density (graphene)
	double u_C_max;		// maximum
	double u_BN_min;	// minimum of smoothed density (HBN)
	double u_BN_max;	// maximum
	
	double *A_C;		// operator for linear part, e^{-k^2 \hat{\mathcal{L}} \Delta t} (graphene)
	double *B_C;		// operator for nonlinear part ...
	double *p_C;		// array for \psi_C
	double *q_C;		// another array
	double *u_C;		// smoothed \psi_C
	double *A_BN;		// operator for linear part (HBN)
	double *B_BN;		// operator for nonlinear part
	double *p_B;		// ...
	double *q_B;
	double *p_N;
	double *q_N;
	double *u_BN;
	double *s;			// substrate
	
	fftw_plan p_P_C;	// FFTW plan for F(p) (graphene)
	fftw_plan q_Q_C;	// F(q)
	fftw_plan Q_q_C;	// F^-1(Q)
	fftw_plan p_P_B;	// ... (boron)
	fftw_plan q_Q_B;	// ...
	fftw_plan Q_q_B;
	fftw_plan p_P_N;
	fftw_plan q_Q_N;
	fftw_plan Q_q_N;
	fftw_plan u_U_C;
	fftw_plan U_u_C;
	fftw_plan u_U_BN;
	fftw_plan U_u_BN;
};

// contains model parameters
struct Model {
	double alpha_C;		// see documentation and/or input file...
	double alpha_BN;
	double alpha_C_B;
	double alpha_B_C;
	double alpha_C_N;
	double alpha_N_C;
	double alpha_B_N;
	double alpha_C_S;	// couplings to substrate
	double alpha_B_S;
	double alpha_N_S;
	double beta_C;
	double beta_BN;
	double beta_B_N;
	double gamma_C_l;
	double gamma_C_s;
	double gamma_BN_l;
	double gamma_BN_s;
	double gamma_B_N;
	double gamma_N_B;
	double gamma_C_B;
	double gamma_B_C;
	double gamma_C_N;
	double gamma_N_C;
	double delta_C;
	double delta_BN;
	double l_BN;
	double sigma_u;
	double sigma_mask;
};

// contains stuff related to relaxation
struct Relaxation {
	time_t t0;			// start time
	int id;				// rank of process
	int ID;				// number of processes
	int t;				// time step count
	double f_C;			// free energy density (graphene)
	double f_BN;		// ... (HBN)
	double p_C;			// average density (graphene)
	double p_BN;
	double d;			// sampling step size for calculation box optimization
	
	int T;				// total number of iterations
	double dx;			// x-discretization
	double dy;			// y-...
	double dt;			// time step
	int T_optimize;		// interval for calculation box optimization
};

// seeds processes' random number generators
void seed_rngs(struct Relaxation *relaxation, FILE *input) {
	int seed;								// seed for next process
	fscanf(input, " %d", &seed);			// seed for 0th process from input file
	if(relaxation->id == 0) {
		if(seed == 0) srand(time(NULL));	// random seed
		else srand(seed);					// user-specified seed
		seed = rand();						// sample new seed for next process
	}
	else {
		MPI_Recv(&seed, 1, MPI_INT, relaxation->id-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	// receive from previous
		srand(seed);																			// seed
		seed = rand();																			// new seed for next
	}
	if(relaxation->id != relaxation->ID-1) MPI_Send(&seed, 1, MPI_INT, relaxation->id+1, 0, MPI_COMM_WORLD);	// send to next
}

// configures arrays and FFTW plans
void configure_arrays(struct Arrays *arrays, FILE *input) {
	fscanf(input, " %d %d", &arrays->W, &arrays->H);

	int lWHh = fftw_mpi_local_size_2d_transposed(arrays->H, arrays->W/2+1, MPI_COMM_WORLD, &arrays->lH, &arrays->lh0, &arrays->lW, &arrays->lw0);		// local data size halved
	int lWHp = 2*lWHh;	// local data size padded

	arrays->A_C = (double*)fftw_malloc(lWHh*sizeof(double));	// allocate arrays
	arrays->B_C = (double*)fftw_malloc(lWHh*sizeof(double));
	arrays->A_BN = (double*)fftw_malloc(lWHh*sizeof(double));
	arrays->B_BN = (double*)fftw_malloc(lWHh*sizeof(double));
	
	arrays->p_C = (double*)fftw_malloc(lWHp*sizeof(double));
	arrays->q_C = (double*)fftw_malloc(lWHp*sizeof(double));
	arrays->p_B = (double*)fftw_malloc(lWHp*sizeof(double));
	arrays->q_B = (double*)fftw_malloc(lWHp*sizeof(double));
	arrays->p_N = (double*)fftw_malloc(lWHp*sizeof(double));
	arrays->q_N = (double*)fftw_malloc(lWHp*sizeof(double));
	
	arrays->u_C = (double*)fftw_malloc(lWHp*sizeof(double));
	arrays->u_BN = (double*)fftw_malloc(lWHp*sizeof(double));
	
	arrays->s = (double*)fftw_malloc(lWHp*sizeof(double));		// fftw_calloc doesn't exist*
	int Wp = 2*(arrays->W/2+1);
	int w, h, k;
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) arrays->s[k+w] = 0.0;	// *substrate set to zero in case not initialized otherwise
	}
	
	arrays->p_P_C = fftw_mpi_plan_dft_r2c_2d(arrays->H, arrays->W, arrays->p_C, (fftw_complex*)arrays->p_C, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);	// set up FFTW plans
	arrays->q_Q_C = fftw_mpi_plan_dft_r2c_2d(arrays->H, arrays->W, arrays->q_C, (fftw_complex*)arrays->q_C, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
	arrays->Q_q_C = fftw_mpi_plan_dft_c2r_2d(arrays->H, arrays->W, (fftw_complex*)arrays->q_C, arrays->q_C, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_IN);
	arrays->p_P_B = fftw_mpi_plan_dft_r2c_2d(arrays->H, arrays->W, arrays->p_B, (fftw_complex*)arrays->p_B, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
	arrays->q_Q_B = fftw_mpi_plan_dft_r2c_2d(arrays->H, arrays->W, arrays->q_B, (fftw_complex*)arrays->q_B, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
	arrays->Q_q_B = fftw_mpi_plan_dft_c2r_2d(arrays->H, arrays->W, (fftw_complex*)arrays->q_B, arrays->q_B, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_IN);
	arrays->p_P_N = fftw_mpi_plan_dft_r2c_2d(arrays->H, arrays->W, arrays->p_N, (fftw_complex*)arrays->p_N, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
	arrays->q_Q_N = fftw_mpi_plan_dft_r2c_2d(arrays->H, arrays->W, arrays->q_N, (fftw_complex*)arrays->q_N, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
	arrays->Q_q_N = fftw_mpi_plan_dft_c2r_2d(arrays->H, arrays->W, (fftw_complex*)arrays->q_N, arrays->q_N, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_IN);
	
	arrays->u_U_C = fftw_mpi_plan_dft_r2c_2d(arrays->H, arrays->W, arrays->u_C, (fftw_complex*)arrays->u_C, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
	arrays->U_u_C = fftw_mpi_plan_dft_c2r_2d(arrays->H, arrays->W, (fftw_complex*)arrays->u_C, arrays->u_C, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_IN);
	arrays->u_U_BN = fftw_mpi_plan_dft_r2c_2d(arrays->H, arrays->W, arrays->u_BN, (fftw_complex*)arrays->u_BN, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
	arrays->U_u_BN = fftw_mpi_plan_dft_c2r_2d(arrays->H, arrays->W, (fftw_complex*)arrays->u_BN, arrays->u_BN, MPI_COMM_WORLD, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_IN);
	
}

// configures output
void configure_output(struct Output *output, FILE *input) {
	fscanf(input, " %d %d", &output->T_print, &output->T_write);	// read intervals for printing and saving from input file
}

// configures model
void configure_model(struct Model *model, FILE *input) {
	fscanf(input, " %lf %lf %lf %lf %lf", &model->alpha_C, &model->beta_C, &model->gamma_C_l, &model->gamma_C_s, &model->delta_C);	// read model parameters from input file
	fscanf(input, " %lf %lf %lf %lf %lf", &model->alpha_BN, &model->beta_BN, &model->gamma_BN_l, &model->gamma_BN_s, &model->delta_BN);
	fscanf(input, " %lf %lf %lf", &model->alpha_B_N, &model->beta_B_N, &model->gamma_B_N);
	fscanf(input, " %lf %lf", &model->alpha_C_B, &model->gamma_C_B);
	fscanf(input, " %lf %lf", &model->alpha_C_N, &model->gamma_C_N);
	fscanf(input, " %lf %lf", &model->alpha_B_C, &model->gamma_B_C);
	fscanf(input, " %lf %lf", &model->alpha_N_C, &model->gamma_N_C);
	fscanf(input, " %lf", &model->l_BN);
	fscanf(input, " %lf %lf", &model->sigma_u, &model->sigma_mask);
	fscanf(input, " %lf %lf %lf", &model->alpha_C_S, &model->alpha_B_S, &model->alpha_N_S);	// substrate couplings
}

// configures relaxation
void configure_relaxation(struct Relaxation *relaxation, FILE *input) {
	fscanf(input, " %d %lf %lf %lf %d", &relaxation->T, &relaxation->dx, &relaxation->dy, &relaxation->dt, &relaxation->T_optimize);	// read from input file
}

// returns the one-mode approximation for close-packed lattice
// x0, y0 give rotation (theta) axis; u, v give translation; l0 dimensionless lattice constant
double OMA(double x0, double y0, double u, double v, double l0, double theta, int fold) {
	double qx, qy;						// wave numbers
	qx = 2.0*pi/l0;
	if(fold == 6) qy = qx/sqrt(3.0);	// close-packed lattice*
	else if(fold == 4) qy = qx;			// rectangular ((100) substrate)**
	else {
		qx = 0.0;						// other lattices not supported
		qy = 0.0;
	}
	double cosine = cos(theta);
	double sine = sin(theta);
	double x = cosine*x0-sine*y0+u;		// rotation matrix applied
	double y = sine*x0+cosine*y0+v;
	if(fold == 6) return cos(qx*x)*cos(qy*y)+0.5*cos(2.0*qy*y);	// *
	else if(fold == 4) return 4.0*cos(qx*x)*cos(qy*y);			// *
	else return 0.0;
}

// initializes system with a random tiled heterostructure
void tiled_state(struct Arrays *arrays, double dx, double dy, double l0, double l_BN, int N, int M, double p_C_l, double p_C_s, double A_C, double p_BN_l, double p_BN_s, double A_BN) {
	int Wp = 2*(arrays->W/2+1);
	int n, m;	// tile indices
	int *phases = (int*)fftw_malloc(N*M*sizeof(int));			// array for indicating graphene/HBN tiles
	double *thetas = (double*)fftw_malloc(N*M*sizeof(double));	// rotations
	int l = 0;	// counter for HBN tiles
	int id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);	// get process's rank
	if(id == 0) {						// 0th process samples tile structure
		for(m = 0; m < M; m++) {
			for(n = 0; n < N; n++) {
				if((double)rand()/RAND_MAX < (0.5*N*M-l)/(N*M-(N*m+n))) {	// tries to keep phase fractions equal (the more there are HBN tiles the more likely graphene ones are and vice versa)
					phases[N*m+n] = 1;						// HBN tile
					l++;									// update counter
				}
				else phases[N*m+n] = 0;						// graphene tile
				thetas[N*m+n] = 2.0*pi*rand()/RAND_MAX;		// random rotation
			}
		}
	}
	MPI_Bcast(phases, N*M, MPI_INT, 0, MPI_COMM_WORLD);		// broadcast tile structure to other processes
	MPI_Bcast(thetas, N*M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	int w, gh, k;
	for(m = 0; m < M; m++) {
		for(n = 0; n < N; n++) {
			if(phases[N*m+n] == 1) {	// HBN tile
				double theta = thetas[N*m+n];
				for(gh = arrays->H/M*m; gh < arrays->H/M*(m+1); gh++) {			// go through tile's global y-indices
					if(gh >= arrays->lh0 && gh < arrays->lh0+arrays->lH) {		// if within local chunk
						k = Wp*(gh-arrays->lh0);	// local row start index
						for(w = arrays->W/N*n; w < arrays->W/N*(n+1); w++) {
							arrays->q_C[k+w] = p_C_l;	// graphene phase disordered
							arrays->q_B[k+w] = p_BN_s+A_BN*OMA(w*dx, gh*dy, 0.5*l_BN*l0, -0.5*l_BN*l0/sqrt(3.0), l_BN*l0, theta, 6);		// HBN phase crystalline
							arrays->q_N[k+w] = p_BN_s+A_BN*OMA(w*dx, gh*dy, 0.5*l_BN*l0, 0.5*l_BN*l0/sqrt(3.0), l_BN*l0, theta, 6);
						}
					}
				}
			}
			else {						// graphene tile
				double theta = thetas[N*m+n];
				for(gh = arrays->H/M*m; gh < arrays->H/M*(m+1); gh++) {
					if(gh >= arrays->lh0 && gh < arrays->lh0+arrays->lH) {
						k = Wp*(gh-arrays->lh0);
						for(w = arrays->W/N*n; w < arrays->W/N*(n+1); w++) {
							arrays->q_C[k+w] = p_C_s+A_C*OMA(w*dx, gh*dy, 0.0, 0.0, l0, theta, 6);
							arrays->q_B[k+w] = p_BN_l;
							arrays->q_N[k+w] = p_BN_l;
						}
					}
				}
			}
		}
	}
	fftw_free(phases);	// free tile arrays
	fftw_free(thetas);
}

// initializes system with a bicrystalline heterostructure
// rotation axes at phase centers: (0, H/2), (W/2, H/2)
void bicrystalline_state(struct Arrays *arrays, double dx, double dy, double l0, double l_BN, double x, double p_C_l, double p_C_s, double A_C, double theta_C, double p_BN_l, double p_BN_s, double A_BN, double theta_BN) {
	int Wp = 2*(arrays->W/2+1);			// padded width
	int w, h, gh, k, v;					// indices
	double deg2rad = pi/180.0;
	theta_C *= deg2rad;
	theta_BN *= deg2rad;
	for(h = 0; h < arrays->lH; h++) {
		gh = arrays->lh0+h;				// global vertical index
		k = Wp*h;						// local row start index
		for(w = 0; w < arrays->W; w++) {
			if(w >= 0.5*(1.0-x)*arrays->W && w < 0.5*(1.0+x)*arrays->W) {	// HBN phase
				arrays->q_C[k+w] = p_C_l;		// set graphene to disordered state
				arrays->q_B[k+w] = p_BN_s+A_BN*OMA((w-0.5*arrays->W)*dx, (gh-0.5*arrays->H)*dy, 0.5*l_BN*l0, -0.5*l_BN*l0/sqrt(3.0), l_BN*l0, theta_BN, 6);				// HBN crystalline
				arrays->q_N[k+w] = p_BN_s+A_BN*OMA((w-0.5*arrays->W)*dx, (gh-0.5*arrays->H)*dy, 0.5*l_BN*l0, 0.5*l_BN*l0/sqrt(3.0), l_BN*l0, theta_BN, 6);
			}
			else {															// graphene phase
				if(w < 0.5*arrays->W) v = w;	// v = w, when w < W/2
				else v = w-arrays->W;			// else v = w-W
				arrays->q_C[k+w] = p_C_s+A_C*OMA(v*dx, (gh-0.5*arrays->H)*dy, 0.0, 0.0, l0, theta_C, 6);	// crystalline
				arrays->q_B[k+w] = p_BN_l;		// disordered
				arrays->q_N[k+w] = p_BN_l;
			}
		}
	}
}

// initializes system with a regular-polygon crystallite of one phase embedded in the other phase serving as a matrix
// Polygon corners (fold gives their number) are a distance R from the center (W/2, H/2), one of the corners has an angle phi wrt. positive x-axis
// use large fold for circular crystallite, e.g. a 1000-gon
void embedded_crystallite(struct Arrays *arrays, double dx, double dy, double l0, double l_BN, int fold, double R, double phi, int phases, double p_C_l, double p_C_s, double A_C, double theta_C, double u_C, double v_C, double p_BN_l, double p_BN_s, double A_BN, double theta_BN, double u_BN, double v_BN) {
	if(fold < 3) {
		printf("The crystallite must be a polygon (fold >= 3)!\n");
		exit(1);
	}
	int Wp = 2*(arrays->W/2+1);			// padded width
	int w, h, gh, k;					// indices
	double deg2rad = pi/180.0;
	phi *= deg2rad;
	theta_C *= deg2rad;
	theta_BN *= deg2rad;
	double corners[fold][2];
	int f;
	for(f = 0; f < fold; f++) {
		corners[f][0] = R*cos(2.0*pi*f/fold+phi);
		corners[f][1] = R*sin(2.0*pi*f/fold+phi);
	}
	double R2 = R*R;
	double Rin2 = R*cos(pi/fold);
	Rin2 *= Rin2;
	double x, y, y2, r2, ax, ay, bx, by, z;
	int in;
	for(h = 0; h < arrays->lH; h++) {
		gh = arrays->lh0+h;				// global vertical index
		y = (gh-0.5*arrays->H)*dy;
		y2 = y*y;
		k = Wp*h;						// local row start index
		for(w = 0; w < arrays->W; w++) {
			x = (w-0.5*arrays->W)*dx;
			r2 = x*x+y2;
			if(r2 < Rin2) in = 1;
			else if(r2 > R2) in = 0;
			else {
				in = 1;
				ax = corners[0][0]-corners[fold-1][0];
				ay = corners[0][1]-corners[fold-1][1];
				bx = x-corners[fold-1][0];
				by = y-corners[fold-1][1];
				z = ax*by-ay*bx;
				for(f = 1; f < fold; f++) {
					ax = corners[f][0]-corners[f-1][0];
					ay = corners[f][1]-corners[f-1][1];
					bx = x-corners[f-1][0];
					by = y-corners[f-1][1];
					if((ax*by-ay*bx)*z < 0.0) {
						in = 0;
						break;
					}
				}
			}
			if(in == 1) {		// crystallite
				if(phases == 0) {	// graphene
					arrays->q_C[k+w] = p_C_s+A_C*OMA((w-0.5*arrays->W)*dx, (gh-0.5*arrays->H)*dy, u_C, v_C, l0, theta_C, 6);
					arrays->q_B[k+w] = p_BN_l;
					arrays->q_N[k+w] = p_BN_l;
				}
				else {				// HBN
					arrays->q_C[k+w] = p_C_l;
					arrays->q_B[k+w] = p_BN_s+A_BN*OMA((w-0.5*arrays->W)*dx, (gh-0.5*arrays->H)*dy, 0.5*l_BN*l0+u_BN, -0.5*l_BN*l0/sqrt(3.0)+v_BN, l_BN*l0, theta_BN, 6);
					arrays->q_N[k+w] = p_BN_s+A_BN*OMA((w-0.5*arrays->W)*dx, (gh-0.5*arrays->H)*dy, 0.5*l_BN*l0+u_BN, 0.5*l_BN*l0/sqrt(3.0)+v_BN, l_BN*l0, theta_BN, 6);
				}
			}
			else {					// matrix
				if(phases == 0) {	// HBN
					arrays->q_C[k+w] = p_C_l;
					arrays->q_B[k+w] = p_BN_s+A_BN*OMA((w-0.5*arrays->W)*dx, (gh-0.5*arrays->H)*dy, 0.5*l_BN*l0+u_BN, -0.5*l_BN*l0/sqrt(3.0)+v_BN, l_BN*l0, theta_BN, 6);
					arrays->q_N[k+w] = p_BN_s+A_BN*OMA((w-0.5*arrays->W)*dx, (gh-0.5*arrays->H)*dy, 0.5*l_BN*l0+u_BN, 0.5*l_BN*l0/sqrt(3.0)+v_BN, l_BN*l0, theta_BN, 6);
				}
				else {				// graphene
					arrays->q_C[k+w] = p_C_s+A_C*OMA((w-0.5*arrays->W)*dx, (gh-0.5*arrays->H)*dy, u_C, v_C, l0, theta_C, 6);
					arrays->q_B[k+w] = p_BN_l;
					arrays->q_N[k+w] = p_BN_l;
				}
			}
		}
	}
}

// initializes system with a state read from a file
// only a single phase (graphene or HBN) is loaded (both can first be initialized in a different way, then one can be overwritten with this)
void load_state(struct Arrays *arrays, char *filename, int which) {
	FILE *file;
	file = fopen(filename, "r");
	int Wp = 2*(arrays->W/2+1);
	int w, h, k;
	double p_C, p_B, p_N, u_C, u_BN, dv, dummy;
	for(h = 0; h < arrays->H; h++) {
		if(h >= arrays->lh0 && h < arrays->lh0+arrays->lH) {	// data corresponds to local chunk
			k = Wp*(h-arrays->lh0);
			for(w = 0; w < arrays->W; w++) {
				fscanf(file, "%lf %lf %lf %lf %lf %lf", &p_C, &p_B, &p_N, &u_C, &u_BN, &dv);
				if(which == 0) arrays->q_C[k+w] = p_C;	// load graphene phase
				else {
					arrays->q_B[k+w] = p_B;				// HBN
					arrays->q_N[k+w] = p_N;
				}
			}
		} else for(w = 0; w < arrays->W; w++) {					// skip other processes data
			fscanf(file, "%lf %lf %lf %lf %lf %lf", &dummy, &dummy, &dummy, &dummy, &dummy, &dummy);
		}
	}
	fclose(file);
}

// initializes system
void initialize_system(struct Arrays *arrays, FILE *input) {
	int type;				// type of initialization
	fscanf(input, " %d", &type);
	if(type == 0) {			// tiled state
		int N, M;
		double dx, dy, l0, l_BN, p_C_l, p_C_s, A_C, p_BN_l, p_BN_s, A_BN;
		fscanf(input, " %lf %lf %lf %lf %d %d %lf %lf %lf %lf %lf %lf", &dx, &dy, &l0, &l_BN, &N, &M, &p_C_l, &p_C_s, &A_C, &p_BN_l, &p_BN_s, &A_BN);
		tiled_state(arrays, dx, dy, l0, l_BN, N, M, p_C_l, p_C_s, A_C, p_BN_l, p_BN_s, A_BN);
	}
	else if(type == 1) {	// bicrystalline state
		double dx, dy, l0, l_BN, x, p_C_l, p_C_s, A_C, theta_C, p_BN_l, p_BN_s, A_BN, theta_BN;
		fscanf(input, " %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &dx, &dy, &l0, &l_BN, &x, &p_C_l, &p_C_s, &A_C, &theta_C, &p_BN_l, &p_BN_s, &A_BN, &theta_BN);
		bicrystalline_state(arrays, dx, dy, l0, l_BN, x, p_C_l, p_C_s, A_C, theta_C, p_BN_l, p_BN_s, A_BN, theta_BN);
	}
	else if(type == 2) {	// embedded crystallite
		int fold, phases;
		double dx, dy, l0, l_BN, R, phi, p_C_l, p_C_s, A_C, theta_C, u_C, v_C, p_BN_l, p_BN_s, A_BN, theta_BN, u_BN, v_BN;
		fscanf(input, " %lf %lf %lf %lf %d %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &dx, &dy, &l0, &l_BN, &fold, &R, &phi, &phases, &p_C_l, &p_C_s, &A_C, &theta_C, &u_C, &v_C, &p_BN_l, &p_BN_s, &A_BN, &theta_BN, &u_BN, &v_BN);
		embedded_crystallite(arrays, dx, dy, l0, l_BN, fold, R, phi, phases, p_C_l, p_C_s, A_C, theta_C, u_C, v_C, p_BN_l, p_BN_s, A_BN, theta_BN, u_BN, v_BN);
	}
	else if(type == 3) {	// load saved state
		int which;
		char filename[128];
		fscanf(input, " %s %d", filename, &which);
		load_state(arrays, filename, which);
	}
}

// initializes substrate with single crystalline state
// only close-packed and rectangular lattice are supported (fold = 6 and 4, respectively)
void substrate(struct Arrays *arrays, int fold, double dx, double dy, double l0, double p, double A, double theta, double u, double v) {
	int Wp = 2*(arrays->W/2+1);
	int w, h, gh, k;
	theta *= pi/180.0;
	for(h = 0; h < arrays->lH; h++) {
		gh = arrays->lh0+h;				// global vertical index
		k = Wp*h;						// local row start index
		for(w = 0; w < arrays->W; w++) arrays->s[k+w] = p+A*OMA((w-0.5*arrays->W)*dx, (gh-0.5*arrays->H)*dy, u, v, l0, theta, fold);
	}
}

// initializes substrate with a state read from a pfc1 data file (or similar, i.e. single-column format)
void load_substrate(struct Arrays *arrays, char *filename) {
	FILE *file;
	file = fopen(filename, "r");
	int Wp = 2*(arrays->W/2+1);
	int w, h, k;
	double p;
	for(h = 0; h < arrays->H; h++) {
		if(h >= arrays->lh0 && h < arrays->lh0+arrays->lH) {	// data corresponds to local chunk
			k = Wp*(h-arrays->lh0);
			for(w = 0; w < arrays->W; w++) {
				fscanf(file, "%lf", &p);
				arrays->s[k+w] = p;
			}
		} else for(w = 0; w < arrays->W; w++) {					// skip other processes data
			fscanf(file, "%lf", &p);
		}
	}
	fclose(file);
}

// initializes substrate
void initialize_substrate(struct Arrays *arrays, FILE *input) {
	int type;
	fscanf(input, " %d", &type);
	if(type == 0) {			// crystalline substrate
		int fold;
		double dx, dy, l0, p, A, theta, u, v;
		fscanf(input, " %d %lf %lf %lf %lf %lf %lf %lf %lf", &fold, &dx, &dy, &l0, &p, &A, &theta, &u, &v);
		substrate(arrays, fold, dx, dy, l0, p, A, theta, u, v);
	}
	else if(type == 1) {	// load saved state as substrate
		char filename[128];
		fscanf(input, " %s", filename);
		load_substrate(arrays, filename);
	}
}

// maps x (min <= x <= max) linearly between 0 and 1
double map(double x, double min, double max) {
	if(min == max) return min;		// avoid divide by zero
	else return (x-min)/(max-min);
}

// maps substrate (min0 <= s <= max0) linearly between min and max
void map_substrate(struct Arrays *arrays, FILE *input) {
	int Wp = 2*(arrays->W/2+1);
	int w, h, k;
	double min0 = 1.0e100;
	double max0 = -1.0e100;
	// determine extrema
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) {
			if(arrays->s[k+w] < min0) min0 = arrays->s[k+w];
			if(arrays->s[k+w] > max0) max0 = arrays->s[k+w];
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &min0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &max0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	double min, max;
	fscanf(input, " %lf %lf", &min, &max);
	// map
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) arrays->s[k+w] = map(arrays->s[k+w], min0, max0)*(max-min)+min;
	}
}

// shifts the average density of the substrate to p and scales its RMS amplitude to A
void shift_scale_substrate(struct Arrays *arrays, FILE *input) {
	int Wp = 2*(arrays->W/2+1);
	int w, h, k;
	double ave0 = 0.0;
	// determine average density
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) ave0 += arrays->s[k+w];
	}
	MPI_Allreduce(MPI_IN_PLACE, &ave0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	ave0 *= 1.0/arrays->W/arrays->H;
	double S = 0.0;
	// set average to zero and compute square sum
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) {
			arrays->s[k+w] -= ave0;
			S += arrays->s[k+w]*arrays->s[k+w];
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &S, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double p, A;
	fscanf(input, " %lf %lf", &p, &A);
	double c = A*sqrt((double)arrays->W*arrays->H/S);
	// scale and shift
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) arrays->s[k+w] = arrays->s[k+w]*c+p;
	}
}

// introduces random noise to initial state
// noise is sampled from uniform distribution
void add_noise(struct Arrays *arrays, FILE *input) {
	double A_C, A_BN;	// separate noise peak amplitudes for both phases
	fscanf(input, " %lf %lf", &A_C, &A_BN);
	int w, h, k;
	int Wp = 2*(arrays->W/2+1);
	double n_C, n_B, n_N;	// noise
	double sum_C = 0.0;		// counters for added noise
	double sum_B = 0.0;
	double sum_N = 0.0;
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) {
			n_C = A_C*(2.0*rand()/RAND_MAX-1.0);	// sample noise
			n_B = A_BN*(2.0*rand()/RAND_MAX-1.0);
			n_N = A_BN*(2.0*rand()/RAND_MAX-1.0);
			arrays->q_C[k+w] += n_C;				// add noise
			arrays->q_B[k+w] += n_B;
			arrays->q_N[k+w] += n_N;
			sum_C += n_C;							// update counters
			sum_B += n_B;
			sum_N += n_N;
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &sum_C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	// communicate
	MPI_Allreduce(MPI_IN_PLACE, &sum_B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &sum_N, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	sum_C /= (double)arrays->W*arrays->H;			// offset per grid point due to noise
	sum_B /= (double)arrays->W*arrays->H;
	sum_N /= (double)arrays->W*arrays->H;
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) {
			arrays->q_C[k+w] -= sum_C;		// restore average densities
			arrays->q_B[k+w] -= sum_B;
			arrays->q_N[k+w] -= sum_N;
		}
	}
}

// saves state into a file, format:
// C field, B field, N field, smoothed density (graphene), smoothed density (HBN), \Delta\phi (see documentation)
void write_state(struct Arrays *arrays, struct Relaxation *relaxation, struct Output *output) {
	char filename[128];			// filename
	sprintf(filename, "%s_t:%d.dat", output->name, relaxation->t);
	FILE *file;					// file stream
	int Wp = 2*(arrays->W/2+1);
	int i, w, h, k;
	double v_C, v_BN, dv;		// normalized smoothed densities, \Delta\phi
	for(i = 0; i < relaxation->ID; i++) {	// go through ranks
		MPI_Barrier(MPI_COMM_WORLD);		// makes every process wait until previous has completed writing output
		if(relaxation->id == i) {			// ith process continues, others go back waiting
			if(relaxation->id == 0) file = fopen(filename, "w");	// 0th process overwrites possible previous data
			else file = fopen(filename, "a");						// others append to end
			for(h = 0; h < arrays->lH; h++) {
				k = Wp*h;
				for(w = 0; w < arrays->W; w++) {
					v_C = map(arrays->u_BN[k+w], arrays->u_BN_max, arrays->u_BN_min);	// normalize smoothed densities
					v_BN = map(arrays->u_C[k+w], arrays->u_C_min, arrays->u_C_max);
					dv = (v_C-v_BN)/(v_C+v_BN);											// \Delta\phi
					fprintf(file, "%e %e %e %e %e %e %e\n", arrays->q_C[k+w], arrays->q_B[k+w], arrays->q_N[k+w], arrays->u_C[k+w], arrays->u_BN[k+w], dv, arrays->s[k+w]);	// substrate last column
				}
			}
			fclose(file);		// close file stream
		}
	}
}

// prints output, format:
// time step count, elapsed wall-clock time (s), dx (dimensionless*), dy (*), average free energy density (total) (*), average free energy density (graphene) (*), average free energy density (HBN) (*), average density (graphene) (*), average density (HBN) (*)
void print(struct Relaxation *relaxation, struct Output *output) {
	if(relaxation->id == 0) {			// only 0th process
		printf("%d %d %lf %lf %lf %lf %lf %lf %lf\n", relaxation->t, (int)(time(NULL)-relaxation->t0), relaxation->dx, relaxation->dy, relaxation->f_C+relaxation->f_BN, relaxation->f_C, relaxation->f_BN, relaxation->p_C, relaxation->p_BN);
		char filename[128];
		sprintf(filename, "%s.out", output->name);
		FILE *file;
		file = fopen(filename, "a");	// need only append, empty file generated in main()
		fprintf(file, "%d %d %lf %lf %lf %lf %lf %lf %lf\n", relaxation->t, (int)(time(NULL)-relaxation->t0), relaxation->dx, relaxation->dy, relaxation->f_C+relaxation->f_BN, relaxation->f_C, relaxation->f_BN, relaxation->p_C, relaxation->p_BN);
		fclose(file);
	}
}

// updates operators for linear and nonlinear parts
void update_AB(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
	int w, h, gw, k;
	double ky, kx2, k2;									// (squared) k-vector (components)
	double divWH = 1.0/arrays->W/arrays->H;				// 1/(W*H)
	double dkx = 2.0*pi/(relaxation->dx*arrays->W);		// discretization in k-space
	double dky = 2.0*pi/(relaxation->dy*arrays->H);
	double d2, l, expl;									// (1+\nabla^2)^2, variables for intermediate results
	double q_BN2 = 1.0/(model->l_BN*model->l_BN);		// squared wavenumber for HBN
	for(w = 0; w < arrays->lW; w++) {
		gw = arrays->lw0+w;								// global x-index
		kx2 = gw*dkx;									// kx
		kx2 *= kx2;										// kx^2
		k = arrays->H*w;
		for(h = 0; h < arrays->H; h++) {
			if(h < arrays->H/2) ky = h*dky;				// upper half negative ky-values
			else ky = (h-arrays->H)*dky;
			k2 = kx2+ky*ky;								// k^2
			// graphene
			d2 = (1.0-k2);
			d2 *= d2;
			l = model->alpha_C+model->beta_C*d2;		// alpha+beta*(1+\nabla^2)^2 in k-space
			expl = exp(-k2*l*relaxation->dt);			// e^{-k^2 \hat{\mathcal{L}} \Delta t}
			arrays->A_C[k+h] = expl;
			if(l == 0.0) arrays->B_C[k+h] = -k2*relaxation->dt;		// avoid divide by zero
			else arrays->B_C[k+h] = (expl-1.0)/l;
			arrays->B_C[k+h] *= divWH;					// descaling
			// HBN
			d2 = (q_BN2-k2);
			d2 *= d2;
			l = model->alpha_BN+model->beta_BN*d2;
			expl = exp(-k2*l*relaxation->dt);
			arrays->A_BN[k+h] = expl;
			if(l == 0.0) arrays->B_BN[k+h] = -k2*relaxation->dt;
			else arrays->B_BN[k+h] = (expl-1.0)/l;
			arrays->B_BN[k+h] *= divWH;
		}
	}
}

// scales p arrays (in k-space) by 1/(W*H) ((I)FFTs cause scaling of data by sqrt(W*H))
void scale_Ps(struct Arrays *arrays) {
	fftw_complex *P_C, *P_B, *P_N;			// complex data pointers
	P_C = (fftw_complex*)&arrays->p_C[0];
	P_B = (fftw_complex*)&arrays->p_B[0];
	P_N = (fftw_complex*)&arrays->p_N[0];
	double divWH = 1.0/arrays->W/arrays->H;
	int i;
	int lA = arrays->lW*arrays->H;
	for(i = 0; i < lA; i++) {
		P_C[i] *= divWH;
		P_B[i] *= divWH;
		P_N[i] *= divWH;
	}
}

// scales q arrays (in k-space) by 1/(W*H)
void scale_Qs(struct Arrays *arrays) {
	fftw_complex *Q_C, *Q_B, *Q_N;
	Q_C = (fftw_complex*)&arrays->q_C[0];
	Q_B = (fftw_complex*)&arrays->q_B[0];
	Q_N = (fftw_complex*)&arrays->q_N[0];
	double divWH = 1.0/arrays->W/arrays->H;
	int i;
	int lA = arrays->lW*arrays->H;
	for(i = 0; i < lA; i++) {
		Q_C[i] *= divWH;
		Q_B[i] *= divWH;
		Q_N[i] *= divWH;
	}
}

// computes average free energy densities and average densities
void fp(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
	int Wp = 2*(arrays->W/2+1);
	double dkx = 2.0*pi/(relaxation->dx*arrays->W);
	double dky = 2.0*pi/(relaxation->dy*arrays->H);
	memcpy(arrays->p_C, arrays->q_C, Wp*arrays->lH*sizeof(double));		// save current state to p arrays
	memcpy(arrays->p_B, arrays->q_B, Wp*arrays->lH*sizeof(double));
	memcpy(arrays->p_N, arrays->q_N, Wp*arrays->lH*sizeof(double));
	fftw_execute(arrays->q_Q_C);
	fftw_execute(arrays->q_Q_B);
	fftw_execute(arrays->q_Q_N);
	scale_Qs(arrays);
	double kx2, ky, k2, d2;
	fftw_complex *Q_C = (fftw_complex*)&arrays->q_C[0];
	fftw_complex *Q_B = (fftw_complex*)&arrays->q_B[0];
	fftw_complex *Q_N = (fftw_complex*)&arrays->q_N[0];
	double q2 = 1.0/(model->l_BN*model->l_BN);
	int w, h, gw, k;
	for(w = 0; w < arrays->lW; w++) {
		gw = arrays->lw0+w;
		kx2 = gw*dkx;
		kx2 *= kx2;
		k = arrays->H*w;
		for(h = 0; h < arrays->H; h++) {
			if(h < arrays->H/2) ky = h*dky;
			else ky = (h-arrays->H)*dky;
			k2 = kx2+ky*ky;
			
			d2 = (1.0-k2);
			d2 *= d2;
			Q_C[k+h] *= d2;		// compute (1-\nabla^2)^2 \psi_C in k-space
			
			d2 = (q2-k2);
			d2 *= d2;
			Q_B[k+h] *= d2;
			Q_N[k+h] *= d2;
		}
	}
	fftw_execute(arrays->Q_q_C);
	fftw_execute(arrays->Q_q_B);
	fftw_execute(arrays->Q_q_N);
	relaxation->f_C = 0.0;		// reset variables for average (free energy) density
	relaxation->f_BN = 0.0;
	relaxation->p_C = 0.0;
	relaxation->p_BN = 0.0;
	double div3 = 1.0/3.0;		// factor 1/3 (multiplication is faster than division (although the compiler may be smart enough to do this itself))
	double a = -1.0/(2.0*model->sigma_mask*model->sigma_mask);	// exponential factor for Gaussian mask
	double p_C, p_B, p_N, q_C, q_B, q_N, s;	// variables to simplify expressions
	double p_C2, p_B2, p_N2, v_C, v_BN, gamma_C, gamma_BN, dv, mask;
	double alpha_C = model->alpha_C;		// variables for model parameters to simplify expressions
	double beta_C = model->beta_C;
	double delta_C = model->delta_C;
	double alpha_C_B = model->alpha_C_B;
	double gamma_C_B = model->gamma_C_B;
	double alpha_C_N = model->alpha_C_N;
	double gamma_C_N = model->gamma_C_N;
	double alpha_BN = model->alpha_BN;
	double beta_BN = model->beta_BN;
	double delta_BN = model->delta_BN;
	double alpha_B_N = model->alpha_B_N;
	double beta_B_N = model->beta_B_N;
	double gamma_B_N = model->gamma_B_N;
	double alpha_B_C = model->alpha_B_C;
	double gamma_B_C = model->gamma_B_C;
	double alpha_N_C = model->alpha_N_C;
	double gamma_N_C = model->gamma_N_C;
	double alpha_C_S = model->alpha_C_S;
	double alpha_B_S = model->alpha_B_S;
	double alpha_N_S = model->alpha_N_S;
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) {
			p_C = arrays->p_C[k+w];
			p_B = arrays->p_B[k+w];
			p_N = arrays->p_N[k+w];
			q_C = arrays->q_C[k+w];
			q_B = arrays->q_B[k+w];
			q_N = arrays->q_N[k+w];
			s = arrays->s[k+w];
			p_C2 = p_C*p_C;
			p_B2 = p_B*p_B;
			p_N2 = p_N*p_N;
			v_C = map(arrays->u_BN[k+w], arrays->u_BN_max, arrays->u_BN_min);			// normalized smoothed densities (1 where crystalline, 0 where disordered)
			v_BN = map(arrays->u_C[k+w], arrays->u_C_min, arrays->u_C_max);
			gamma_C = v_C*(model->gamma_C_s-model->gamma_C_l)+model->gamma_C_l;			// "chemical potential fields"
			gamma_BN = v_BN*(model->gamma_BN_s-model->gamma_BN_l)+model->gamma_BN_l;
			dv = (v_C-v_BN)/(v_C+v_BN);													// \Delta \phi (ideally 1 for crystalline graphene, -1 for crystalline HBN)
			mask = exp(a*dv*dv);														// mask function
			// average free energy density (graphene)
			relaxation->f_C += 	0.5*alpha_C*p_C2	+ 0.5*beta_C*p_C*q_C				+ div3*gamma_C*p_C*p_C2					+ 0.25*delta_C*p_C2*p_C2
			+ mask*(
								alpha_C_B*p_C*p_B 					  					+ 0.5*gamma_C_B*( p_C2*p_B + p_C*p_B2 )
							  + alpha_C_N*p_C*p_N					  					+ 0.5*gamma_C_N*( p_C2*p_N + p_C*p_N2 )
					)
			+ v_C*alpha_C_S*p_C*s;							// contribution from couplings to substrate*, v_C and v_BN (from 0 to 1) used to mask coupling
			// (HBN)
			relaxation->f_BN += 0.5*alpha_BN*p_B2	+ 0.5*beta_BN*p_B*q_B				+ div3*gamma_BN*p_B*p_B2				+ 0.25*delta_BN*p_B2*p_B2
							  + 0.5*alpha_BN*p_N2	+ 0.5*beta_BN*p_N*q_N				+ div3*gamma_BN*p_N*p_N2				+ 0.25*delta_BN*p_N2*p_N2
							  + alpha_B_N*p_B*p_N	+ beta_B_N*p_B*q_N					+ 0.5*gamma_B_N*( p_B2*p_N + p_B*p_N2 )
			+ mask*(
							    alpha_B_C*p_B*p_C										+ 0.5*gamma_B_C*( p_B2*p_C + p_B*p_C2 )
							  + alpha_N_C*p_N*p_C										+ 0.5*gamma_N_C*( p_N2*p_C + p_N*p_C2 )
					)
			+ v_BN*( alpha_B_S*p_B + alpha_N_S*p_N )*s;		// *
			// restore q arrays
			arrays->q_C[k+w] = p_C;
			arrays->q_B[k+w] = p_B;
			arrays->q_N[k+w] = p_N;
			// average densities
			relaxation->p_C += p_C;
			relaxation->p_BN += 0.5*(p_B+p_N);
		}
	}
	// communication between processes, note that can't reduce as average because chunks may have different sizes!
	MPI_Allreduce(MPI_IN_PLACE, &relaxation->f_C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &relaxation->f_BN, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &relaxation->p_C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &relaxation->p_BN, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double divWH = 1.0/arrays->W/arrays->H;
	relaxation->f_C *= divWH;		// divide to get averages
	relaxation->f_BN *= divWH;
	relaxation->p_C *= divWH;
	relaxation->p_BN *= divWH;
	fftw_execute(arrays->p_P_C);	// restore p arrays in k-space
	fftw_execute(arrays->p_P_B);
	fftw_execute(arrays->p_P_N);
	scale_Ps(arrays);
}

// updates smoothed density fields
void update_us(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
	int Wp = 2*(arrays->W/2+1);
	int w, gw, h, k;
	double divWH = 1.0/arrays->W/arrays->H;
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) {
			arrays->u_C[k+w] = arrays->q_C[k+w];
			arrays->u_BN[k+w] = 0.5*(arrays->q_B[k+w]+arrays->q_N[k+w]);	// average of B and N fields
		}
	}
	fftw_execute(arrays->u_U_C);
	fftw_execute(arrays->u_U_BN);
	double ky, kx2, k2;
	double a = -1.0/(2.0*model->sigma_u*model->sigma_u);	// exponential factor for Gaussian smoothing
	double dkx = 2.0*pi/(relaxation->dx*arrays->W);
	double dky = 2.0*pi/(relaxation->dy*arrays->H);
	fftw_complex *U_C = (fftw_complex*)&arrays->u_C[0];
	fftw_complex *U_BN = (fftw_complex*)&arrays->u_BN[0];
	for(w = 0; w < arrays->lW; w++) {
		gw = arrays->lw0+w;
		kx2 = gw*dkx;
		kx2 *= kx2;
		k = arrays->H*w;
		for(h = 0; h < arrays->H; h++) {
			if(h < arrays->H/2) ky = h*dky;
			else ky = (h-arrays->H)*dky;
			k2 = kx2+ky*ky;
			U_C[k+h] *= exp(a*k2)*divWH;		// Gaussian convolution in k-space
			U_BN[k+h] *= exp(a*k2)*divWH;
		}
	}
	fftw_execute(arrays->U_u_C);
	fftw_execute(arrays->U_u_BN);
	arrays->u_C_min = 1.0e100;		// reset variables for density field extrema
	arrays->u_C_max = -1.0e100;
	arrays->u_BN_min = 1.0e100;
	arrays->u_BN_max = -1.0e100;
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) {
			if(arrays->u_C[k+w] < arrays->u_C_min) arrays->u_C_min = arrays->u_C[k+w];		// determine density field extrema
			if(arrays->u_C[k+w] > arrays->u_C_max) arrays->u_C_max = arrays->u_C[k+w];
			if(arrays->u_BN[k+w] < arrays->u_BN_min) arrays->u_BN_min = arrays->u_BN[k+w];
			if(arrays->u_BN[k+w] > arrays->u_BN_max) arrays->u_BN_max = arrays->u_BN[k+w];
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &arrays->u_C_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);	// communication
	MPI_Allreduce(MPI_IN_PLACE, &arrays->u_C_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &arrays->u_BN_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &arrays->u_BN_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

// performs one iteration of the semi-implicit spectral method
void step(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
	int w, h, k;
	int Wp = 2*(arrays->W/2+1);
	if(relaxation->t%100 == 0) {	// avoids numerical instability (I've never understood why the instability occurs but this fixes it)
		memcpy(arrays->p_C, arrays->q_C, Wp*arrays->lH*sizeof(double));
		memcpy(arrays->p_B, arrays->q_B, Wp*arrays->lH*sizeof(double));
		memcpy(arrays->p_N, arrays->q_N, Wp*arrays->lH*sizeof(double));
		fftw_execute(arrays->p_P_C);
		fftw_execute(arrays->p_P_B);
		fftw_execute(arrays->p_P_N);
		scale_Ps(arrays);
	}
	double v_C, v_BN, gamma_C, gamma_BN, dv, mask;
	double a = -1.0/(2.0*model->sigma_mask*model->sigma_mask);
	double q_C, q_B, q_N, s, q_C2, q_B2, q_N2;
	double delta_C = model->delta_C;
	double alpha_C_B = model->alpha_C_B;
	double gamma_C_B = model->gamma_C_B;
	double alpha_C_N = model->alpha_C_N;
	double gamma_C_N = model->gamma_C_N;
	double delta_BN = model->delta_BN;
	double alpha_B_N = model->alpha_B_N;
	double beta_B_N = model->beta_B_N;
	double gamma_B_N = model->gamma_B_N;
	double alpha_B_C = model->alpha_B_C;
	double gamma_B_C = model->gamma_B_C;
	double alpha_N_C = model->alpha_N_C;
	double gamma_N_C = model->gamma_N_C;
	double alpha_C_S = model->alpha_C_S;
	double alpha_B_S = model->alpha_B_S;
	double alpha_N_S = model->alpha_N_S;
	for(h = 0; h < arrays->lH; h++) {
		k = Wp*h;
		for(w = 0; w < arrays->W; w++) {
			v_C = map(arrays->u_BN[k+w], arrays->u_BN_max, arrays->u_BN_min);
			v_BN = map(arrays->u_C[k+w], arrays->u_C_min, arrays->u_C_max);
			gamma_C = v_C*( model->gamma_C_s - model->gamma_C_l ) + model->gamma_C_l;
			gamma_BN = v_BN*( model->gamma_BN_s - model->gamma_BN_l ) + model->gamma_BN_l;
			dv = (v_C-v_BN)/(v_C+v_BN);
			mask = exp(a*dv*dv);
			q_C = arrays->q_C[k+w];
			q_B = arrays->q_B[k+w];
			q_N = arrays->q_N[k+w];
			s = arrays->s[k+w];
			q_C2 = q_C*q_C;
			q_B2 = q_B*q_B;
			q_N2 = q_N*q_N;
			// compute nonlinear part (C)
			arrays->q_C[k+w] = gamma_C*q_C2 + delta_C*q_C*q_C2 + mask*( alpha_C_B*q_B + gamma_C_B*( q_C*q_B + 0.5*q_B2 ) + alpha_C_N*q_N + gamma_C_N*( q_C*q_N + 0.5*q_N2 ) ) + v_C*alpha_C_S*s;	// substrate couplings at end
			// (B)
			arrays->q_B[k+w] = gamma_BN*q_B2 + delta_BN*q_B*q_B2 + alpha_B_N*q_N + gamma_B_N*( q_B*q_N + 0.5*q_N2 ) + mask*( alpha_B_C*q_C + gamma_B_C*( q_B*q_C + 0.5*q_C2 ) ) + v_BN*alpha_B_S*s;
			// (N)
			arrays->q_N[k+w] = gamma_BN*q_N2 + delta_BN*q_N*q_N2 + alpha_B_N*q_B + gamma_B_N*( q_N*q_B + 0.5*q_B2 ) + mask*( alpha_N_C*q_C + gamma_N_C*( q_N*q_C + 0.5*q_C2 ) ) + v_BN*alpha_N_S*s;
		}
	}
	fftw_execute(arrays->q_Q_C);
	fftw_execute(arrays->q_Q_B);
	fftw_execute(arrays->q_Q_N);
	int gw;
	double WH = (double)arrays->W*arrays->H;
	double dkx = 2.0*pi/(relaxation->dx*arrays->W);
	double dky = 2.0*pi/(relaxation->dy*arrays->H);
	double kx2, ky, k2, d2;
	double q2 = 1.0/(model->l_BN*model->l_BN);
	fftw_complex P_B_prev;
	fftw_complex *P_C = (fftw_complex*)&arrays->p_C[0];
	fftw_complex *P_B = (fftw_complex*)&arrays->p_B[0];
	fftw_complex *P_N = (fftw_complex*)&arrays->p_N[0];
	fftw_complex *Q_C = (fftw_complex*)&arrays->q_C[0];
	fftw_complex *Q_B = (fftw_complex*)&arrays->q_B[0];
	fftw_complex *Q_N = (fftw_complex*)&arrays->q_N[0];
	for(w = 0; w < arrays->lW; w++) {
		gw = arrays->lw0+w;
		kx2 = gw*dkx;
		kx2 *= kx2;
		k = arrays->H*w;
		for(h = 0; h < arrays->H; h++) {
			P_B_prev = P_B[k+h];			// P_B[k+h] will be overwritten so its current value is saved for computing P_N[k+h]
			if(h < arrays->H/2) ky = h*dky;
			else ky = (h-arrays->H)*dky;
			k2 = kx2+ky*ky;
			// update \psi_C in k-space
			P_C[k+h] = arrays->A_C[k+h]*P_C[k+h] + arrays->B_C[k+h]*Q_C[k+h];
			// \psi_B and \psi_N
			d2 = q2-k2;
			d2 *= d2;
			P_B[k+h] = arrays->A_BN[k+h]*P_B[k+h] + arrays->B_BN[k+h]*( Q_B[k+h] + beta_B_N*d2*P_N[k+h]*WH );
			P_N[k+h] = arrays->A_BN[k+h]*P_N[k+h] + arrays->B_BN[k+h]*( Q_N[k+h] + beta_B_N*d2*P_B_prev*WH );
			// substrate is static so it's not updated like the other fields
			// copy p's to q's
			Q_C[k+h] = P_C[k+h];
			Q_B[k+h] = P_B[k+h];
			Q_N[k+h] = P_N[k+h];
		}
	}
	fftw_execute(arrays->Q_q_C);
	fftw_execute(arrays->Q_q_B);
	fftw_execute(arrays->Q_q_N);
	update_us(arrays, model, relaxation);	// update smoothed density fields
}

// optimizes calculation box size with system size
// samples slightly different sizes and interpolates optimal size
void optimize(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
	// sample sizes
	double dxs[] = {relaxation->dx, relaxation->dx-relaxation->d, relaxation->dx+relaxation->d, relaxation->dx, relaxation->dx};
	double dys[] = {relaxation->dy, relaxation->dy, relaxation->dy, relaxation->dy-relaxation->d, relaxation->dy+relaxation->d};
	double fs[5];	// average free energy densities of sizes sampled
	int i;
	for(i = 0; i < 5; i++) {		// sample sizes
		relaxation->dx = dxs[i];
		relaxation->dy = dys[i];
		fp(arrays, model, relaxation);				// compute energy (and density (not needed here though))
		fs[i] = relaxation->f_C+relaxation->f_BN;	// save total average free energy density
	}
	// interpolate dx and dy (minimum of surface f = a+b*dx+c*dy+d*dx^2+e*dy^2)
	relaxation->dx = (relaxation->d*(fs[1]-fs[2])+2.0*dxs[0]*(-2.0*fs[0]+fs[1]+fs[2]))/(2.0*(fs[1]-2.0*fs[0]+fs[2]));
	relaxation->dy = (relaxation->d*(fs[3]-fs[4])+2.0*dys[0]*(-2.0*fs[0]+fs[3]+fs[4]))/(2.0*(fs[3]-2.0*fs[0]+fs[4]));
	// check that change in discretization is acceptable
	double l0 = 4.0*pi/sqrt(3.0);						// approximate dimensionless lattice constant
	double dw = arrays->W*(relaxation->dx-dxs[0]);		// change in horizontal system size
	double dh = arrays->H*(relaxation->dy-dys[0]);		// ... vertical ...
	double dr = sqrt(dw*dw+dh*dh);						// change vector
	double x = 0.25*l0;									// limit to 1/4 of lattice constant (to ensure stability)
	if(dr > x) {	// if the change in system dimensions exceeds 1/4 of the lattice constant ...
		x /= dr;
		relaxation->dx = x*relaxation->dx+(1.0-x)*dxs[0];	// ... truncate the change to 1/4 of the lattice constant
		relaxation->dy = x*relaxation->dy+(1.0-x)*dys[0];
	}
	// update A and B
	update_AB(arrays, model, relaxation);	// dx and dy changed -> need to update operators
	// update sampling step size (tries to keep it on par with dr (hopefully more accurate optimization))
	double ddx = relaxation->dx-dxs[0];
	double ddy = relaxation->dy-dys[0];
	double ddr = sqrt(ddx*ddx+ddy*ddy);					// discretization change
	if(ddr < relaxation->d) relaxation->d *= 0.5;		// if change vector < d, halve d
	else relaxation->d *= 2.0;							// otherwise double
	if(relaxation->d < 1.0e-6) relaxation->d *= 2.0;	// can cause numerical issues if d gets too small
}

// relaxes the system for T time steps
void relax(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation, struct Output *output) {
	for(relaxation->t = 0; relaxation->t <= relaxation->T; relaxation->t++) {
		if(relaxation->T_optimize > 0 && relaxation->t > 0 && relaxation->t%relaxation->T_optimize == 0) optimize(arrays, model, relaxation);	// optimize?
		if(relaxation->t%output->T_print == 0) {	// print output?
			fp(arrays, model, relaxation);
			print(relaxation, output);
		}
		if(relaxation->t%output->T_write == 0) write_state(arrays, relaxation, output);		// save state?
		if(relaxation->t < relaxation->T) step(arrays, model, relaxation);	// perform iteration step
	}
}

// frees allocated arrays
void clear_arrays(struct Arrays *arrays) {
	fftw_free(arrays->p_C);
	fftw_free(arrays->q_C);
	fftw_free(arrays->p_B);
	fftw_free(arrays->q_B);
	fftw_free(arrays->p_N);
	fftw_free(arrays->q_N);
	fftw_free(arrays->u_C);
	fftw_free(arrays->u_BN);
	fftw_free(arrays->A_C);
	fftw_free(arrays->B_C);
	fftw_free(arrays->A_BN);
	fftw_free(arrays->B_BN);
	fftw_free(arrays->s);		// substrate freed
}

int main(int argc, char **argv) {
	// init MPI
	MPI_Init(&argc, &argv);
	fftw_mpi_init();
	// create structs
	struct Arrays arrays;
	struct Model model;
	struct Relaxation relaxation;
	struct Output output;
	relaxation.t0 = time(NULL);
	relaxation.t = 0;
	relaxation.d = 0.0001;
	MPI_Comm_rank(MPI_COMM_WORLD, &relaxation.id);
	MPI_Comm_size(MPI_COMM_WORLD, &relaxation.ID);
	strcpy(output.name, argv[1]);
	// input stream
	char filename[128];
	sprintf(filename, "%s.in", output.name);
	FILE *input = fopen(filename, "r");
	if(input == NULL) {
		printf("Error: Input file not found!\n");
		return 1;
	}
	// create empty output file
	sprintf(filename, "%s.out", output.name);
	FILE *out = fopen(filename, "w");
	fclose(out);
	// read input
	char label;
	char line[1024];
	while(fscanf(input, " %c", &label) != EOF) {
		if(label == '#' && fgets(line, 1024, input)) {}			// comment
		else if(label == 'S') {									// seed random number generators
			seed_rngs(&relaxation, input);
		}
		else if(label == 'O') {									// output
			configure_output(&output, input);
		}
		else if(label == 'A') {									// set up arrays and FFTW plans
			configure_arrays(&arrays, input);
		}
		else if(label == 'I') {									// initialization
			initialize_system(&arrays, input);
		}
		else if(label == 'N') {									// add noise
			add_noise(&arrays, input);
		}
		else if(label == 'M') {									// model
			configure_model(&model, input);
		}
		else if(label == 'R') {									// relaxation
			configure_relaxation(&relaxation, input);
			update_AB(&arrays, &model, &relaxation);
			update_us(&arrays, &model, &relaxation);
			relax(&arrays, &model, &relaxation, &output);
		}
		else if(label == 's') {									// configure substrate
			if(fscanf(input, " %c", &label) == EOF) return 1;
			if(label == 'I') {									// initialize substrate
				initialize_substrate(&arrays, input);
			}
			else if(label == 'M') {								// map substrate
				map_substrate(&arrays, input);
			}
			else if(label == 'S') {								// shift and scale (RMS) substrate
				shift_scale_substrate(&arrays, input);
			}
			else {
				printf("Invalid label: %c!\n", label);			// bad input
				return 1;
			}
		}
		else {
			printf("Invalid label: %c!\n", label);				// bad input
			return 1;
		}
	}
	fclose(input);
	
	clear_arrays(&arrays);										// clean-up
	MPI_Finalize();

	return 0;

}
