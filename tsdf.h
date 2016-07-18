#include "tsdf_kernels.cu"

class Scene3D{
public:
  std::string filename;
  std::string seqname;

  float K[9];
  float R[9];
  unsigned int width;
  unsigned int height;
  unsigned int len_pcIndex;
  unsigned int len_beIndex;
  unsigned int len_beLinIdx;
  std::vector<Box3D> objects;
  std::vector<Box2D> objects_2d_tight;
  std::vector<Box2D> objects_2d_full;

  bool GPUdata;
  Scene3DType DataType;
  // defined in .data file
  unsigned int* grid_range;
  float* begin_range;
  float grid_delta;
  RGBDpixel* RGBDimage;
  unsigned int* beIndex;
  unsigned int* beLinIdx;
  unsigned int* pcIndex;
  float* XYZimage;
  float* K_GPU;
  float* R_GPU;

  Scene3D(){
      RGBDimage = NULL;
      beIndex = NULL;
      pcIndex = NULL;
      beLinIdx = NULL;
      XYZimage = NULL;
      grid_range = NULL;
      begin_range = NULL;
      K_GPU = NULL;
      R_GPU = NULL;

      GPUdata = false;
      DataType = RGBD;
  };
  void compute_xyzGPU() {
      if (!GPUdata){
         std::cout<< "Data is not at GPU cannot compute_xyz at GPU"<<std::endl;
         FatalError(__LINE__);
      }
      if (XYZimage!=NULL){
         std::cout<< "XYZimage!=NULL"<<std::endl;
         FatalError(__LINE__);
      }
      checkCUDA(__LINE__, cudaMalloc(&XYZimage, sizeof(float)*width*height*3));
      compute_xyzkernel<<<width,height>>>(XYZimage,RGBDimage,K_GPU,R_GPU);
  }

  void loadData2XYZimage(){
      switch(DataType){
        case RGBD:
            {
              this ->load();
              this -> cpu2gpu();
              this -> compute_xyzGPU();
            }
            break;
        }
  };


  int load(){
    int filesize =0;
    if (RGBDimage==NULL||beIndex==NULL||pcIndex==NULL||XYZimage==NULL){
      free();
      FILE* fp = fopen(filename.c_str(),"rb");
      if (fp==NULL) { std::cout<<"in load() :fail to open file: "<<filename<<std::endl; exit(EXIT_FAILURE); }
      grid_range = new unsigned int[3];
      filesize += fread((void*)(grid_range), sizeof(unsigned int), 3, fp);
      
      begin_range = new float[3];
      filesize += fread((void*)(begin_range), sizeof(float), 3, fp);
      filesize += fread((void*)(&grid_delta), sizeof(float), 1, fp);

      RGBDimage = new RGBDpixel[width*height];
      filesize += fread((void*)(RGBDimage), sizeof(RGBDpixel), width*height, fp);

      filesize +=  fread((void*)(&len_beIndex), sizeof(unsigned int), 1, fp);
      beIndex   = new unsigned int [len_beIndex];
      filesize += fread((void*)(beIndex), sizeof(unsigned int), len_beIndex, fp);

      filesize +=  fread((void*)(&len_beLinIdx), sizeof(unsigned int), 1, fp);
      beLinIdx  = new unsigned int [len_beLinIdx];
      filesize += fread((void*)(beLinIdx), sizeof(unsigned int), len_beLinIdx, fp);

      filesize += fread((void*)(&len_pcIndex), sizeof(unsigned int), 1, fp);
      pcIndex   = new unsigned int [len_pcIndex];
      filesize += fread((void*)(pcIndex), sizeof(unsigned int), len_pcIndex, fp);
      fclose(fp);

      GPUdata = false;
      
    }
    return filesize;
  };
  void cpu2gpu(){
    if (!GPUdata){
       if (beIndex!=NULL){
           unsigned int* beIndexCPU = beIndex;
           checkCUDA(__LINE__, cudaMalloc(&beIndex, sizeof(unsigned int)*len_beIndex));
           checkCUDA(__LINE__, cudaMemcpy(beIndex, beIndexCPU,sizeof(unsigned int)*len_beIndex, cudaMemcpyHostToDevice));
           delete [] beIndexCPU;
       }
       else{
           std::cout << "beIndex is NULL"<<std::endl;
       }

       if (beLinIdx!=NULL){
           unsigned int* beLinIdxCPU = beLinIdx;
           checkCUDA(__LINE__, cudaMalloc(&beLinIdx, sizeof(unsigned int)*len_beLinIdx));
           checkCUDA(__LINE__, cudaMemcpy(beLinIdx, beLinIdxCPU,sizeof(unsigned int)*len_beLinIdx, cudaMemcpyHostToDevice));
           delete [] beLinIdxCPU;
       }
       else{
           std::cout << "beLinIdx is NULL"<<std::endl;
       }

       // make it to full matrix to skip searching 
       unsigned int * beIndexFull;
       unsigned int sz = 2*sizeof(unsigned int)*(grid_range[0]+1)*(grid_range[1]+1)*(grid_range[2]+1);
       checkCUDA(__LINE__, cudaMalloc(&beIndexFull, sz));
       checkCUDA(__LINE__, cudaMemset(beIndexFull, 0, sz));
       int THREADS_NUM = 1024;
       int BLOCK_NUM = int((len_beLinIdx + size_t(THREADS_NUM) - 1) / THREADS_NUM);
       fillInBeIndexFull<<<BLOCK_NUM,THREADS_NUM>>>(beIndexFull,beIndex,beLinIdx,len_beLinIdx);
       checkCUDA(__LINE__,cudaGetLastError());
       checkCUDA(__LINE__, cudaFree(beIndex));      beIndex = NULL;
       checkCUDA(__LINE__, cudaFree(beLinIdx));     beLinIdx = NULL;
       beIndex = beIndexFull;

       if (pcIndex!=NULL){
          unsigned int* pcIndexCPU = pcIndex;
          checkCUDA(__LINE__, cudaMalloc(&pcIndex, sizeof(unsigned int)*len_pcIndex));
          checkCUDA(__LINE__, cudaMemcpy(pcIndex, pcIndexCPU,sizeof(unsigned int)*len_pcIndex, cudaMemcpyHostToDevice));
          delete [] pcIndexCPU;
       }
       else{
           std::cout << "pcIndexCPU is NULL"<<std::endl;
       }
       

       if (RGBDimage!=NULL){
         RGBDpixel* RGBDimageCPU = RGBDimage;
         checkCUDA(__LINE__, cudaMalloc(&RGBDimage, sizeof(RGBDpixel)*width*height));
         checkCUDA(__LINE__, cudaMemcpy( RGBDimage, RGBDimageCPU, sizeof(RGBDpixel)*width*height, cudaMemcpyHostToDevice));
         delete [] RGBDimageCPU;
       }
       else{
           std::cout << "RGBDimage is NULL"<<std::endl;
       }

       if (grid_range!=NULL){ 
          unsigned int * grid_rangeCPU = grid_range;
          checkCUDA(__LINE__, cudaMalloc(&grid_range, sizeof(unsigned int)*3));
          checkCUDA(__LINE__, cudaMemcpy(grid_range, grid_rangeCPU, 3*sizeof(unsigned int), cudaMemcpyHostToDevice));
          delete [] grid_rangeCPU;
       }
       else{
          std::cout << "grid_range is NULL"<<std::endl;
       }

       if (begin_range!=NULL){ 
          float * begin_rangeCPU = begin_range;
          checkCUDA(__LINE__, cudaMalloc(&begin_range, sizeof(float)*3));
          checkCUDA(__LINE__, cudaMemcpy(begin_range, begin_rangeCPU, sizeof(float)*3, cudaMemcpyHostToDevice));
          delete [] begin_rangeCPU;
       }
       else{
          std::cout << "grid_range is NULL"<<std::endl;
       }


       checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(K_GPU, (float*)K, sizeof(float)*9, cudaMemcpyHostToDevice));

      
       checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(R_GPU, (float*)R, sizeof(float)*9, cudaMemcpyHostToDevice)); 

       GPUdata = true;

    }
  };

  void free(){
    if (GPUdata){
      
      if (RGBDimage   !=NULL) {checkCUDA(__LINE__, cudaFree(RGBDimage));    RGBDimage = NULL;}
      if (beIndex     !=NULL) {checkCUDA(__LINE__, cudaFree(beIndex));      beIndex = NULL;}
      if (beLinIdx    !=NULL) {checkCUDA(__LINE__, cudaFree(beLinIdx));     beLinIdx = NULL;}
      if (pcIndex     !=NULL) {checkCUDA(__LINE__, cudaFree(pcIndex));      pcIndex = NULL;}
      if (XYZimage    !=NULL) {checkCUDA(__LINE__, cudaFree(XYZimage));     XYZimage = NULL;}
      if (R_GPU       !=NULL) {checkCUDA(__LINE__, cudaFree(R_GPU));        R_GPU = NULL;}
      if (K_GPU       !=NULL) {checkCUDA(__LINE__, cudaFree(K_GPU));        K_GPU = NULL;}
      if (grid_range  !=NULL) {checkCUDA(__LINE__, cudaFree(grid_range));   grid_range = NULL;}
      if (begin_range !=NULL) {checkCUDA(__LINE__, cudaFree(begin_range));  begin_range = NULL;}
      GPUdata = false;
    }
    else{
      //std::cout<< "free CPUdata"<<std::endl;
      if (RGBDimage   !=NULL) {delete [] RGBDimage;    RGBDimage   = NULL;}
      if (beIndex     !=NULL) {delete [] beIndex;      beIndex     = NULL;}
      if (beLinIdx    !=NULL) {delete [] beLinIdx;     beLinIdx    = NULL;}
      if (pcIndex     !=NULL) {delete [] pcIndex;      pcIndex     = NULL;}
      if (XYZimage    !=NULL) {delete [] XYZimage;     XYZimage    = NULL;}
      if (grid_range  !=NULL) {delete [] grid_range;   grid_range  = NULL;}
      if (begin_range !=NULL) {delete [] begin_range;  begin_range = NULL;}
    }
  };
  ~Scene3D(){
    free();
  };
};



void compute_TSDF(std::vector<Scene3D*> *chosen_scenes_ptr, std::vector<int> *chosen_box_id, StorageT* datamem, std::vector<int> grid_size, int encode_type, float scale) {
    int totalcounter = 0;
    float tsdf_size = grid_size[1];
  

    int numeltsdf = grid_size[0]*tsdf_size*tsdf_size*tsdf_size;
    int THREADS_NUM = 1024;
    int BLOCK_NUM = int((tsdf_size*tsdf_size*tsdf_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
    float* bb3d_data;

    checkCUDA(__LINE__, cudaMalloc(&bb3d_data,  sizeof(float)*15));
    
  
    for (int sceneId = 0;sceneId<(*chosen_scenes_ptr).size();sceneId++){        
        Scene3D* scene = (*chosen_scenes_ptr)[sceneId];
       
        scene->loadData2XYZimage(); 

        
        int boxId = (*chosen_box_id)[sceneId];
        checkCUDA(__LINE__, cudaMemcpy(bb3d_data, scene->objects[boxId].base, sizeof(float)*15, cudaMemcpyHostToDevice));

        unsigned int * grid_range = scene->grid_range;
        float* R_data = scene->R_GPU;
        float* K_data = scene->K_GPU;
        float* range  = scene->begin_range;
        
        RGBDpixel* RGBDimage = scene->RGBDimage;
        unsigned int* star_end_indx_data = scene->beIndex;
        unsigned int* pc_lin_indx_data = scene->pcIndex;
        float* XYZimage  = scene->XYZimage;
        
        // output
        StorageT * tsdf_data = &datamem[totalcounter*numeltsdf];

         if (encode_type > 99){
            compute_TSDFGPUbox<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data, R_data, K_data, range, scene->grid_delta, grid_range, RGBDimage, 
                           star_end_indx_data, pc_lin_indx_data, XYZimage, bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
                           scene->width, scene->height, encode_type, scale);

        }
        else{
          compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data, R_data, K_data, RGBDimage, XYZimage,
                                                             bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
                                                             scene->width, scene->height, encode_type, scale);
        }
        
        checkCUDA(__LINE__,cudaDeviceSynchronize());
        checkCUDA(__LINE__,cudaGetLastError());

        ++totalcounter;

        //scene_prev = scene;
	scene -> free();
    }
    checkCUDA(__LINE__, cudaFree(bb3d_data));
    

}

