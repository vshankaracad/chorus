
#include "MF_Solver2D_Yee.h"

#include "ElectroMagn.h"
#include "Field2D.h"
// fm
#include "ElectroMagn2D.h"

MF_Solver2D_Yee::MF_Solver2D_Yee( Params &params )
    : Solver2D( params )
{
    isEFilterApplied = params.Friedman_filter;
}

MF_Solver2D_Yee::~MF_Solver2D_Yee()
{
    // EMPTY
}

void MF_Solver2D_Yee::operator()( ElectroMagn *fields )
{

    // const unsigned int nx_p = fields->dimPrim[0];
    const unsigned int nx_d = fields->dimDual[0];
    const unsigned int ny_p = fields->dimPrim[1];
    const unsigned int ny_d = fields->dimDual[1];

    const double *const __restrict__ Ex2D = isEFilterApplied ? fields->filter_->Ex_[0]->data() :
                                                               fields->Ex_->data(); // [x * ny_p + y] : dual in x   primal in y,z
    const double *const __restrict__ Ey2D = isEFilterApplied ? fields->filter_->Ey_[0]->data() :
                                                               fields->Ey_->data(); // [x * ny_d + y] : dual in y   primal in x,z
    const double *const __restrict__ Ez2D = fields->Ez_->data();                    // [x * ny_p + y] : dual in z   primal in x,y
    double *const __restrict__ Bx2D       = fields->Bx_->data();                    // [x * ny_d + y] : dual in y,z primal in x
    double *const __restrict__ By2D       = fields->By_->data();                    // [x * ny_p + y] : dual in x,z primal in y
    double *const __restrict__ Bz2D       = fields->Bz_->data();                    // [x * ny_d + y] : dual in x,y primal in z
   
    //fieldmask
    int i_glob    = ( static_cast<ElectroMagn2D *>( fields ) )->i_glob_;
    int j_glob    = ( static_cast<ElectroMagn2D *>( fields ) )->j_glob_;
    float dx = ( static_cast<ElectroMagn2D *>( fields ) )->dx;
    float dy = ( static_cast<ElectroMagn2D *>( fields ) )->dy;

 
    // Magnetic field Bx^(p,d)
#if defined( SMILEI_ACCELERATOR_GPU_OACC )                                                                                                     
    const int sizeofEx = fields->Ex_->number_of_points_;                                                                               
    const int sizeofEy = fields->Ey_->number_of_points_;                                                                               
    const int sizeofEz = fields->Ez_->number_of_points_;                                                                               
    const int sizeofBx = fields->Bx_->number_of_points_;
    const int sizeofBy = fields->By_->number_of_points_;                                                                               
    const int sizeofBz = fields->Bz_->number_of_points_;                                                                               
    #pragma acc parallel present( Bx2D[0:sizeofBx], Ez2D[0:sizeofEz] )                                               
    #pragma acc loop gang                
#elif defined( SMILEI_ACCELERATOR_GPU_OMP )
    #pragma omp target
    #pragma omp teams distribute parallel for collapse( 2 )
#endif
    for( unsigned int x = 0; x < nx_d - 1; ++x ) {
        int ix_glob = i_glob + x;
        float x_glob = ix_glob * dx;
#if !defined( SMILEI_ACCELERATOR_GPU )
        #pragma omp simd
#endif
#ifdef SMILEI_ACCELERATOR_GPU_OACC                                                                                                             
            #pragma acc loop vector                                                                                                    
#endif
        for( unsigned int y = 1; y < ny_d - 1; ++y ) {
            int iy_glob = j_glob + y;
            float y_glob = iy_glob * dy;
            float fm = 0.0;
            float mcf = 0.01;
            if (x_glob >= 0.0 && x_glob <= 48.0 && y_glob >= 128.0 && y_glob <= 256.0){             
            float taper_x = (x_glob-48.0) / 48.0;   
            float taper_y = (y_glob - 128.0) / 128.0; 
            fm =(1.0 - mcf * mcf * taper_y * taper_y * taper_x * taper_x);
            }
            else { fm=1.0;  }
            Bx2D[x * ny_d + y] -= dt_ov_dy * ( Ez2D[x * ny_p + y] - Ez2D[x * ny_p + y - 1] );
            //Bx2D[x * ny_d + y] *= fm;
            //Bx2D[x * ny_d + y] =(Bx2D[x * ny_d + y]- dt_ov_dy * ( Ez2D[x * ny_p + y] - Ez2D[x * ny_p + y - 1] ))*fm;
        }
    }
    // Magnetic field By^(d,p)
#if defined( SMILEI_ACCELERATOR_GPU_OACC )
    #pragma acc parallel present( By2D[0:sizeofBy], Ez2D[0:sizeofEz] )
    #pragma acc loop gang
#elif defined( SMILEI_ACCELERATOR_GPU_OMP )
    #pragma omp target
    #pragma omp teams distribute parallel for collapse( 2 )
#endif
    for( unsigned int x = 1; x < nx_d - 1; ++x ) {
        int ix_glob = i_glob + x;
        float x_glob = ix_glob * dx;
#if !defined( SMILEI_ACCELERATOR_GPU )
        #pragma omp simd
#endif
#ifdef SMILEI_ACCELERATOR_GPU_OACC                                                                                                             
            #pragma acc loop vector                                                                                                    
#endif
        for( unsigned int y = 0; y < ny_p; ++y ) {
            int iy_glob = j_glob + y;
            float y_glob = iy_glob * dy;
            float fm = 0.0;
            float mcf = 0.01;
            if (x_glob >= 0.0 && x_glob <= 48.0 && y_glob >= 128.0 && y_glob <= 256.0){             
            float taper_x = x_glob / 48.0;   
            float taper_y = (y_glob - 128.0) / 128.0; 
            fm =(1.0 - mcf * mcf * taper_y * taper_y * taper_x * taper_x);
            }
            else { fm=1.0;  }
            By2D[x * ny_p + y] += dt_ov_dx * ( Ez2D[x * ny_p + y] - Ez2D[( x - 1 ) * ny_p + y] );
            //By2D[x * ny_p + y] *= fm;
            //By2D[x * ny_p + y] = (By2D[x * ny_p + y]+dt_ov_dx * ( Ez2D[x * ny_p + y] - Ez2D[( x - 1 ) * ny_p + y] ))*fm;
        }
    }

    // Magnetic field Bz^(d,d)
#if defined( SMILEI_ACCELERATOR_GPU_OACC )
    #pragma acc parallel present( Bz2D[0:sizeofBy], Ex2D[0:sizeofEx], Ey2D[0:sizeofEz] )
    #pragma acc loop gang
#elif defined( SMILEI_ACCELERATOR_GPU_OMP )
    #pragma omp target
    #pragma omp teams distribute parallel for collapse( 2 )
#endif
    for( unsigned int x = 1; x < nx_d - 1; ++x ) {
        int ix_glob = i_glob + x;
        float x_glob = ix_glob * dx;
#if !defined( SMILEI_ACCELERATOR_GPU )
        #pragma omp simd
#endif
#ifdef SMILEI_ACCELERATOR_GPU_OACC                                                                                                             
            #pragma acc loop vector                                                                                                    
#endif 
        for( unsigned int y = 1; y < ny_d - 1; ++y ) {
            int iy_glob = j_glob + y;
            float y_glob = iy_glob * dy;
            float fm = 0.0;
            float mcf = 0.1;

	    if (x_glob >= 0.0 && x_glob <= 15.0 && y_glob >= 128.0 && y_glob <= 256.0){
            float taper_x = (x_glob - 15.0) / 15.0;
            float taper_y = (y_glob - 128.0) / 128.0;
            fm =(1.0 - mcf * mcf * taper_y * taper_y * taper_x * taper_x);
            }
            else if (x_glob >= 0.0 && x_glob <= 15.0 && y_glob >= 0.0 && y_glob <= 128.0) {
            float taper_x = (x_glob - 15.0) / 15.0;
            float taper_y = (y_glob - 128.0) / 128.0;
            fm =(1.0 - mcf * mcf * taper_y * taper_y * taper_x * taper_x); }

	    else if (x_glob >= 0.0 && x_glob <= 48.0 && y_glob >= 236.0 && y_glob <= 256.0) {
            float taper_x = (x_glob - 48.0) / 48.0;
            float taper_y = (y_glob - 236.0) / 20.0;
            fm =(1.0 - mcf * mcf * taper_y * taper_y * taper_x * taper_x); }

            else if (x_glob >= 0.0 && x_glob <= 48.0 && y_glob >= 0.0 && y_glob <= 20.0) {
            float taper_x = (x_glob - 48.0) / 48.0;
            float taper_y = (y_glob - 20.0) / 20.0;
            fm =(1.0 - mcf * mcf * taper_y * taper_y * taper_x * taper_x); }
            else { fm=1.0;  }
            Bz2D[x * ny_d + y] += dt_ov_dy * ( Ex2D[x * ny_p + y] - Ex2D[x * ny_p + y - 1] ) -
                                  dt_ov_dx * ( Ey2D[x * ny_d + y] - Ey2D[( x - 1 ) * ny_d + y] );
            Bz2D[x * ny_d + y] *= fm;
           //MESSAGE("x_glob="<<x_glob<<", fm2D="<<fm<<", x="<<x<<", nx_d="<<nx_d);
           //Bz2D[x * ny_d + y] = (Bz2D[x * ny_d + y]+dt_ov_dy * ( Ex2D[x * ny_p + y] - Ex2D[x * ny_p + y - 1] ) -
           //                       dt_ov_dx * ( Ey2D[x * ny_d + y] - Ey2D[( x - 1 ) * ny_d + y] ))*fm;
        }
    }
}
