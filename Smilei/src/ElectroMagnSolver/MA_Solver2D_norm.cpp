
#include "MA_Solver2D_norm.h"

#include "ElectroMagn.h"
#include "Field2D.h"
// fm
#include "ElectroMagn2D.h"

MA_Solver2D_norm::MA_Solver2D_norm( Params &params )
    : Solver2D( params )
{
    // EMPTY
}

MA_Solver2D_norm::~MA_Solver2D_norm()
{
    // EMPTY
}

void MA_Solver2D_norm::operator()( ElectroMagn *fields )
{
    double *const __restrict__ Ex2D       = fields->Ex_->data(); // [x * ny_p + y] : dual in x   primal in y,z
    double *const __restrict__ Ey2D       = fields->Ey_->data(); // [x * ny_d + y] : dual in y   primal in x,z
    double *const __restrict__ Ez2D       = fields->Ez_->data(); // [x * ny_p + y] : dual in z   primal in x,y
    const double *const __restrict__ Bx2D = fields->Bx_->data(); // [x * ny_d + y] : dual in y,z primal in x
    const double *const __restrict__ By2D = fields->By_->data(); // [x * ny_p + y] : dual in x,z primal in y
    const double *const __restrict__ Bz2D = fields->Bz_->data(); // [x * ny_d + y] : dual in x,y primal in z
    const double *const __restrict__ Jx2D = fields->Jx_->data(); // [x * ny_p + y] : dual in x   primal in y,z
    const double *const __restrict__ Jy2D = fields->Jy_->data(); // [x * ny_d + y] : dual in y   primal in x,z
    const double *const __restrict__ Jz2D = fields->Jz_->data(); // [x * ny_p + y] : dual in z   primal in x,y

    const unsigned int nx_p = fields->dimPrim[0];
    const unsigned int nx_d = fields->dimDual[0];
    const unsigned int ny_p = fields->dimPrim[1];
    const unsigned int ny_d = fields->dimDual[1];

    //fieldmask
    int i_glob    = ( static_cast<ElectroMagn2D *>( fields ) )->i_glob_;
    int j_glob    = ( static_cast<ElectroMagn2D *>( fields ) )->j_glob_;
    float dx = ( static_cast<ElectroMagn2D *>( fields ) )->dx;
    float dy = ( static_cast<ElectroMagn2D *>( fields ) )->dy;

 
    // double sumJx = 0;
    // double sumJy = 0;
    // double sumJz = 0;

    // Electric field Ex^(d,p)
#if defined( SMILEI_ACCELERATOR_GPU_OACC )                                                                                                     
    const int sizeofEx = fields->Ex_->number_of_points_;                                                                               
    const int sizeofEy = fields->Ey_->number_of_points_;                                                                               
    const int sizeofEz = fields->Ez_->number_of_points_;                                                                               
    const int sizeofBx = fields->Bx_->number_of_points_;                                                                               
    const int sizeofBy = fields->By_->number_of_points_;                                                                               
    const int sizeofBz = fields->Bz_->number_of_points_;

    #pragma acc parallel present( Ex2D[0:sizeofEx], Jx2D[0:sizeofEx], Bz2D[0:sizeofBz] )                             
    #pragma acc loop gang
#elif defined( SMILEI_ACCELERATOR_GPU_OMP )
    #pragma omp target
    #pragma omp teams distribute parallel for collapse( 2 )
#endif
    for( unsigned int x = 0; x < nx_d; ++x ) {
        int ix_glob = i_glob + x;
        float x_glob = ix_glob * dx;
#ifdef SMILEI_ACCELERATOR_GPU_OACC
        #pragma acc loop worker
#endif
#if !defined( SMILEI_ACCELERATOR_GPU )
        #pragma omp simd
#endif
        for( unsigned int y = 0; y < ny_p; ++y ) {
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
            else { fm=1.0; }
            Ex2D[x * ny_p + y] += -dt * Jx2D[x * ny_p + y] + dt_ov_dy * ( Bz2D[x * ny_d + y + 1] - Bz2D[x * ny_d + y] );
            Ex2D[x * ny_p + y] *= fm;
            //Ex2D[x * ny_p + y] =(Ex2D[x * ny_p + y]-dt * Jx2D[x * ny_p + y] + dt_ov_dy * ( Bz2D[x * ny_d + y + 1] - Bz2D[x * ny_d + y] ))*fm;
        }
    }

    // Electric field Ey^(p,d)
#if defined( SMILEI_ACCELERATOR_GPU_OACC ) 
    #pragma acc parallel present( Ey2D[0:sizeofEy], Jy2D[0:sizeofEy], Bz2D[0:sizeofBz] )
    #pragma acc loop gang
#elif defined( SMILEI_ACCELERATOR_GPU_OMP )
    #pragma omp target
    #pragma omp teams distribute parallel for collapse( 2 )
#endif
    for( unsigned int x = 0; x < nx_p; ++x ) {
        int ix_glob = i_glob + x;
        float x_glob = ix_glob * dx;
#ifdef SMILEI_ACCELERATOR_GPU_OACC
        #pragma acc loop worker
#endif
#if !defined( SMILEI_ACCELERATOR_GPU )
        #pragma omp simd
#endif
        for( unsigned int y = 0; y < ny_d; ++y ) {
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

            else { fm=1.0; }
            Ey2D[x * ny_d + y] += -dt * Jy2D[x * ny_d + y] - dt_ov_dx * ( Bz2D[( x + 1 ) * ny_d + y] - Bz2D[x * ny_d + y] );
            Ey2D[x * ny_d + y] *= fm;
            //Ey2D[x * ny_d + y] =(Ey2D[x * ny_d + y] - dt * Jy2D[x * ny_d + y] - dt_ov_dx * ( Bz2D[( x + 1 ) * ny_d + y] - Bz2D[x * ny_d + y] ))*fm;
        }
    }

    // Electric field Ez^(p,p)
#if defined( SMILEI_ACCELERATOR_GPU_OACC )
    #pragma acc parallel present( Ez2D[0:sizeofEz], Jz2D[0:sizeofEz], Bx2D[0:sizeofBx], By2D[0:sizeofBy] )                             
    #pragma acc loop gang   
#elif defined( SMILEI_ACCELERATOR_GPU_OMP )
    #pragma omp target
    #pragma omp teams distribute parallel for collapse( 2 )
#endif
    for( unsigned int x = 0; x < nx_p; ++x ) {
        int ix_glob = i_glob + x;
        float x_glob = ix_glob * dx;
#ifdef SMILEI_ACCELERATOR_GPU_OACC
        #pragma acc loop worker
#endif
#if !defined( SMILEI_ACCELERATOR_GPU )
        #pragma omp simd
#endif
        for( unsigned int y = 0; y < ny_p; ++y ) {
            int iy_glob = j_glob + y;
            float y_glob = iy_glob * dy;
            float fm = 0.0;
            float mcf = 0.1;
            if (x_glob >= 0.0 && x_glob <= 48.0 && y_glob >= 200.0 && y_glob <= 256.0){             
            float taper_x = x_glob / 48.0;   
            float taper_y = (y_glob - 200.0) / 56.0; 
            fm =(1.0 - mcf * mcf * taper_y * taper_y * taper_x * taper_x);
            }
            else { fm=1.0;  }
            
            Ez2D[x * ny_p + y] += -dt * Jz2D[x * ny_p + y] +
                                  dt_ov_dx * ( By2D[( x + 1 ) * ny_p + y] - By2D[x * ny_p + y] ) -
                                  dt_ov_dy * ( Bx2D[x * ny_d + y + 1] - Bx2D[x * ny_d + y] );
            //Ez2D[x * ny_p + y] *= fm;
            //Ez2D[x * ny_p + y] =(Ez2D[x * ny_p + y] -dt * Jz2D[x * ny_p + y] +
            //                      dt_ov_dx * ( By2D[( x + 1 ) * ny_p + y] - By2D[x * ny_p + y] ) -
            //                      dt_ov_dy * ( Bx2D[x * ny_d + y + 1] - Bx2D[x * ny_d + y] ))*fm;
        }
    }
}
