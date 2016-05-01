#include <iostream>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <sys/time.h>
using namespace std;

#define StorageT float
#define ComputeT float
#define CPUStorage2ComputeT(x) (x)
#define CPUCompute2StorageT(x) (x)
#define GPUStorage2ComputeT(x) (x)
#define GPUCompute2StorageT(x) (x)

enum Scene3DType { RGBD, Render, Mesh };

struct RGBDpixel{
  uint8_t R;
  uint8_t G;
  uint8_t B;
  uint8_t D;
  uint8_t D_;
};
struct Box3D{
  unsigned int category;
  float base[9];
  float center[3];
  float coeff[3];
};

struct Box2D{
  unsigned int category;
  float tblr[4];
};

unsigned long long get_timestamp_dss(){
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (unsigned long long)now.tv_sec * 1000000;
};

void FatalError(const int lineNumber=0) {
	std::cerr << "FatalError";
	if (lineNumber!=0) std::cerr<<" at LINE "<<lineNumber;
	std::cerr << ". Program Terminated." << std::endl;
	cudaDeviceReset();
	exit(EXIT_FAILURE);
}

void checkCUDA(const int lineNumber, cudaError_t status) {
	if (status != cudaSuccess) {
		std::cerr << "CUDA failure at LINE " << lineNumber << ": " << status << std::endl;
		FatalError();
	}
}

Box3D processbox(Box3D box,float context_pad,int tsdf_size){
     if (context_pad > 0){
        float context_scale = float(tsdf_size) / (float(tsdf_size) - 2*context_pad);
        box.coeff[0] = box.coeff[0] * context_scale;
        box.coeff[1] = box.coeff[1] * context_scale;
        box.coeff[2] = box.coeff[2] * context_scale;
     }
     // change the oreintation 
     if (box.base[1]<0){
        box.base[0] = -1*box.base[0];
        box.base[1] = -1*box.base[1];
        box.base[2] = -1*box.base[2];
     }
     if (box.base[4]<0){
        box.base[3] = -1*box.base[3];
        box.base[4] = -1*box.base[4];
        box.base[5] = -1*box.base[5];
     }

     if(box.base[1]<box.base[4]){
        // swap first two row 
        float tmpbase[3];
        tmpbase[0] = box.base[0];
        tmpbase[1] = box.base[1];
        tmpbase[2] = box.base[2];

        box.base[0] = box.base[3];
        box.base[1] = box.base[4];
        box.base[2] = box.base[5];

        box.base[3] = tmpbase[0];
        box.base[4] = tmpbase[1];
        box.base[5] = tmpbase[2];
        float tmpcoeff =  box.coeff[0];
        box.coeff[0] = box.coeff[1];
        box.coeff[1] = tmpcoeff;
     }
     return box;
}

void __global__ compute_xyzkernel(float * XYZimage, RGBDpixel * RGBDimage, float * K, float * R){
            int ix = blockIdx.x;
            int iy = threadIdx.x;
            int height = blockDim.x;
            //
            uint16_t D = (uint16_t)RGBDimage[iy + ix * height].D;
            uint16_t D_ = (uint16_t)RGBDimage[iy + ix * height].D_;
            D_ = D_<<8;
            float depth = float(D|D_)/1000.0;
            
            // project the depth point to 3d
            float tdx = (float(ix + 1) - K[2]) * depth / K[0];
            float tdz =  - (float(iy + 1) - K[5]) * depth / K[4];
            float tdy = depth;

            XYZimage[3 * (iy + ix * height) + 0] = R[0] * tdx + R[1] * tdy + R[2] * tdz;
            XYZimage[3 * (iy + ix * height) + 1] = R[3] * tdx + R[4] * tdy + R[5] * tdz;
            XYZimage[3 * (iy + ix * height) + 2] = R[6] * tdx + R[7] * tdy + R[8] * tdz;

}

void __global__ fillInBeIndexFull(unsigned int* beIndexFull, unsigned int* beIndex, unsigned int* beLinIdx, unsigned int len_beLinIdx){
     const int index = threadIdx.x + blockIdx.x * blockDim.x;
     if (index>=len_beLinIdx) {
        return;
     }
     else{
        beIndexFull[2*beLinIdx[index]+0] =  beIndex[2*index+0];
        beIndexFull[2*beLinIdx[index]+1] =  beIndex[2*index+1];
     }
}

__global__ void compute_TSDFGPUbox(StorageT* tsdf_data, float* R_data, float* K_data,  float* range, float grid_delta,  unsigned int *grid_range,
                                  RGBDpixel* RGBDimage,  unsigned int* star_end_indx_data ,unsigned int*  pc_lin_indx_data,float* XYZimage,
                                  const float* bb3d_data, int tsdf_size,int tsdf_size1,int tsdf_size2, int fdim, int im_w, int im_h, const int encode_type,const float scale)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;;
    int volume_size = tsdf_size * tsdf_size1 * tsdf_size2;
    if (index > volume_size) return;
    float delta_x = 2 * bb3d_data[12] / float(tsdf_size);  
    float delta_y = 2 * bb3d_data[13] / float(tsdf_size1);  
    float delta_z = 2 * bb3d_data[14] / float(tsdf_size2);  
    float surface_thick = 0.1;
    const float MaxDis = surface_thick + 20;
    //printf("delta_x:%f,%f,%f\n",R_data[0],R_data[1],R_data[2]); 
    // caculate tsdf for this box
    /*
    float x = float(index % tsdf_size);
    float y = float((index / tsdf_size) % tsdf_size);   
    float z = float((index / tsdf_size / tsdf_size) % tsdf_size);
    */
    float x = float((index / (tsdf_size1*tsdf_size2))%tsdf_size) ;
    float y = float((index / tsdf_size2) % tsdf_size1);
    float z = float(index % tsdf_size2);

    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = GPUCompute2StorageT(ComputeT(0));
    }

    // get grid world coordinate
    float temp_x = - bb3d_data[12] + (x + 0.5) * delta_x;
    float temp_y = - bb3d_data[13] + (y + 0.5) * delta_y;
    float temp_z = - bb3d_data[14] + (z + 0.5) * delta_z;

    x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
        + bb3d_data[9];
    y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
        + bb3d_data[10];
    z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
        + bb3d_data[11]; 

    // project to image plane decides the sign
    // rotate back and swap y, z and -y
    float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
    int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

    
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001){
        return;
    } 

    // find the most nearby point 
    float disTosurfaceMin = MaxDis;
    int idx_min = 0;
    int x_grid = floor((x-range[0])/grid_delta);
    int y_grid = floor((y-range[1])/grid_delta);
    int z_grid = floor((z-range[2])/grid_delta);
    //grid_range =  [w,d,h];  linearInd =x(i)*d*h+y(i)*h+z(i);
    //if (x_grid < 0 || x_grid >= grid_range[0] || y_grid < 0 || y_grid >= grid_range[1] || z_grid < 0 || z_grid >= grid_range[2]){
    if (x_grid < 0 || x_grid > grid_range[0] || y_grid < 0 || y_grid > grid_range[1] || z_grid < 0 || z_grid > grid_range[2]){
        return;
    }
    int linearInd =x_grid*grid_range[1]*grid_range[2]+y_grid*grid_range[2]+z_grid;      
    int search_region =1;
    if (star_end_indx_data[2*linearInd+0]>0){
        search_region =0;
    }  
    int find_close_point = -1;

    while(find_close_point<0&&search_region<3){
      for (int iix = max(0,x_grid-search_region); iix < min((int)grid_range[0],x_grid+search_region+1); iix++){
        for (int iiy = max(0,y_grid-search_region); iiy < min((int)grid_range[1],y_grid+search_region+1); iiy++){
          for (int iiz = max(0,z_grid-search_region); iiz < min((int)grid_range[2],z_grid+search_region+1); iiz++){
              unsigned int iilinearInd = iix*grid_range[1]*grid_range[2] + iiy*grid_range[2] + iiz;

              for (int pid = star_end_indx_data[2*iilinearInd+0]-1; pid < star_end_indx_data[2*iilinearInd+1]-1;pid++){
                 
                 //printf("%d-%d\n",star_end_indx_data[2*iilinearInd+0],star_end_indx_data[2*iilinearInd+1]);
                 unsigned int p_idx_lin = pc_lin_indx_data[pid];
                 float xp = XYZimage[3*p_idx_lin+0];
                 float yp = XYZimage[3*p_idx_lin+1];
                 float zp = XYZimage[3*p_idx_lin+2];
                 // distance
                 float xd = abs(x - xp);
                 float yd = abs(y - yp);
                 float zd = abs(z - zp);
                 if (xd < 2.0 * delta_x||yd < 2.0 * delta_x|| zd < 2.0 * delta_x){
                    float disTosurface = sqrt(xd * xd + yd * yd + zd * zd);
                    if (disTosurface < disTosurfaceMin){
                       disTosurfaceMin = disTosurface;
                       idx_min = p_idx_lin;
                       find_close_point = 1;
                       //printf("x:%f,%f,%f,xp,%f,%f,%f,xd%f,%f,%f,%f\n",x,y,z,xp,yp,zp,xd,yd,zd,disTosurfaceMin);
                       
                    }
                }
              } // for all points in this grid
            

          }
        }
      }
      search_region ++;
    }//while 
    
    float tsdf_x = MaxDis;
    float tsdf_y = MaxDis;
    float tsdf_z = MaxDis;


    float color_b =0;
    float color_g =0;
    float color_r =0;

    float xnear = 0;
    float ynear = 0;
    float znear = 0;
    if (find_close_point>0){
        
        xnear = XYZimage[3*idx_min+0];
        ynear = XYZimage[3*idx_min+1];
        znear = XYZimage[3*idx_min+2];
        tsdf_x = abs(x - xnear);
        tsdf_y = abs(y - ynear);
        tsdf_z = abs(z - znear);

        color_b = float(RGBDimage[idx_min].B)/255.0;
        color_g = float(RGBDimage[idx_min].G)/255.0;
        color_r = float(RGBDimage[idx_min].R)/255.0;

        //printf("x:%f,tsdf_x:%f,%f,%f\n",disTosurfaceMin,tsdf_x,tsdf_y,tsdf_z);          
    }
//printf("before : %f,%f,%f\n",tsdf_x,tsdf_y,tsdf_z);

    disTosurfaceMin = min(disTosurfaceMin/surface_thick,float(1.0));
    float ratio = 1.0 - disTosurfaceMin;
    float second_ratio =0;
    if (ratio > 0.5) {
       second_ratio = 1 - ratio;
    }
    else{
       second_ratio = ratio;
    }

    if (disTosurfaceMin > 0.999){
        tsdf_x = MaxDis;
        tsdf_y = MaxDis;
        tsdf_z = MaxDis;
    }

    
    if (encode_type == 101){ 
      tsdf_x = min(tsdf_x, surface_thick);
      tsdf_y = min(tsdf_y, surface_thick);
      tsdf_z = min(tsdf_z, surface_thick);
    }
    else{
      tsdf_x = min(tsdf_x, float(2.0 * delta_x));
      tsdf_y = min(tsdf_y, float(2.0 * delta_y));
      tsdf_z = min(tsdf_z, float(2.0 * delta_z));
    }

   

    float depth_project   = XYZimage[3*(ix * im_h + iy)+1];  
    if (zz > depth_project) {
      tsdf_x = - tsdf_x;
      tsdf_y = - tsdf_y;
      tsdf_z = - tsdf_z;
      disTosurfaceMin = - disTosurfaceMin;
      second_ratio = - second_ratio;
    }

    // encode_type 
    if (encode_type == 100||encode_type == 101){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(tsdf_x);
      tsdf_data[index + 1 * volume_size] = GPUCompute2StorageT(tsdf_y);
      tsdf_data[index + 2 * volume_size] = GPUCompute2StorageT(tsdf_z);
    }
    else if(encode_type == 102){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(tsdf_x);
      tsdf_data[index + 1 * volume_size] = GPUCompute2StorageT(tsdf_y);
      tsdf_data[index + 2 * volume_size] = GPUCompute2StorageT(tsdf_z);
      tsdf_data[index + 3 * volume_size] = GPUCompute2StorageT(color_b/scale);
      tsdf_data[index + 4 * volume_size] = GPUCompute2StorageT(color_g/scale);
      tsdf_data[index + 5 * volume_size] = GPUCompute2StorageT(color_r/scale);
    }
    else if(encode_type == 103){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(ratio);
    }

    // scale feature 
    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = GPUCompute2StorageT(scale* GPUStorage2ComputeT(tsdf_data[index + i * volume_size]));
    }

    //}// end for each index in each box
};


__global__ void compute_TSDFGPUbox_proj(StorageT* tsdf_data, float* R_data, float* K_data, RGBDpixel* RGBDimage, float* XYZimage,
                                      const float* bb3d_data, int tsdf_size,int tsdf_size1,int tsdf_size2, int fdim, int im_w, int im_h, const int encode_type,const float scale)
{
  const int index = threadIdx.x + blockIdx.x * blockDim.x;;
    int volume_size = tsdf_size * tsdf_size1 * tsdf_size2;
    if (index > volume_size) return;
    float delta_x = 2 * bb3d_data[12] / float(tsdf_size);  
    float delta_y = 2 * bb3d_data[13] / float(tsdf_size1);  
    float delta_z = 2 * bb3d_data[14] / float(tsdf_size2);  
    float surface_thick = 0.1;
    const float MaxDis = surface_thick + 20;

    float x = float((index / (tsdf_size1*tsdf_size2))%tsdf_size) ;
    float y = float((index / tsdf_size2) % tsdf_size1);
    float z = float(index % tsdf_size2);

    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = GPUCompute2StorageT(ComputeT(0));
    }

    // get grid world coordinate
    float temp_x = - bb3d_data[12] + (x + 0.5) * delta_x;
    float temp_y = - bb3d_data[13] + (y + 0.5) * delta_y;
    float temp_z = - bb3d_data[14] + (z + 0.5) * delta_z;

    x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
        + bb3d_data[9];
    y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
        + bb3d_data[10];
    z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
        + bb3d_data[11]; 

    // project to image plane decides the sign
    // rotate back and swap y, z and -y
    float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
    int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

    
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001){
        return;
    } 
    
    float x_project   = XYZimage[3*(ix * im_h + iy)+0];
    float y_project   = XYZimage[3*(ix * im_h + iy)+1];
    float z_project   = XYZimage[3*(ix * im_h + iy)+2]; 


    float tsdf_x = abs(x - x_project);
    float tsdf_y = abs(y - y_project);
    float tsdf_z = abs(z - z_project);

    float color_b = 0;
    float color_g = 0;
    float color_r = 0;
    if (RGBDimage!=NULL){
      color_b = float(RGBDimage[(ix * im_h + iy)].B)/255.0;
      color_g = float(RGBDimage[(ix * im_h + iy)].G)/255.0;
      color_r = float(RGBDimage[(ix * im_h + iy)].R)/255.0;
    }

    float disTosurfaceMin = sqrt(tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z);
    disTosurfaceMin = min(disTosurfaceMin/surface_thick,float(1.0));
    float ratio = 1.0 - disTosurfaceMin;
    float second_ratio =0;
    if (ratio > 0.5) {
       second_ratio = 1 - ratio;
    }
    else{
       second_ratio = ratio;
    }
    if (disTosurfaceMin > 0.999){
        tsdf_x = MaxDis;
        tsdf_y = MaxDis;
        tsdf_z = MaxDis;
    }

    tsdf_x = min(tsdf_x, float(2.0 * delta_x));
    tsdf_y = min(tsdf_y, float(2.0 * delta_y));
    tsdf_z = min(tsdf_z, float(2.0 * delta_z));

    if (zz > y_project) {
      tsdf_x = - tsdf_x;
      tsdf_y = - tsdf_y;
      tsdf_z = - tsdf_z;
      disTosurfaceMin = - disTosurfaceMin;
      second_ratio = - second_ratio;
    }

    // encode_type 
    if (encode_type == 0){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(tsdf_x);
      tsdf_data[index + 1 * volume_size] = GPUCompute2StorageT(tsdf_y);
      tsdf_data[index + 2 * volume_size] = GPUCompute2StorageT(tsdf_z);
    }
    if (encode_type == 2){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(tsdf_x);
      tsdf_data[index + 1 * volume_size] = GPUCompute2StorageT(tsdf_y);
      tsdf_data[index + 2 * volume_size] = GPUCompute2StorageT(tsdf_z);
      tsdf_data[index + 3 * volume_size] = GPUCompute2StorageT(color_b/scale);
      tsdf_data[index + 4 * volume_size] = GPUCompute2StorageT(color_g/scale);
      tsdf_data[index + 5 * volume_size] = GPUCompute2StorageT(color_r/scale);
    }
    // scale feature 
    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = GPUCompute2StorageT(scale* GPUStorage2ComputeT(tsdf_data[index + i * volume_size]));
    }
}

