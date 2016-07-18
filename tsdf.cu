#include "tsdf.h"

std::vector<Scene3D*> scenes;
std::vector<int> box_id;
int totalObjectCount = 0;
float scale = 100;
float context_pad =3;
std::vector<int> grid_size {3,208,208,100};
int encode_type =100;
int totalScenes = 0;
string file_list = "boxes_NYU_po_test_nb2000_fb.list";

int main(int argc, char **argv){

	int requestedScene = atoi(argv[1]);
	

	FILE* fp = NULL;
	cout << "Loading file: " << file_list << endl << endl;
	fp = fopen(file_list.c_str(),"rb");
	if (fp==NULL) { cout << "Failed to open file: "<< file_list << endl; exit(EXIT_FAILURE); }


	while (feof(fp)==0)
	{
		Scene3D* scene = new Scene3D();
		unsigned int len = 0;
		fread((void*)(&len), sizeof(unsigned int), 1, fp);    
		if (len==0) return -1;
		scene->filename.resize(len);
		if (len>0) fread((void*)(scene->filename.data()), sizeof(char), len, fp);

		
		string s = scene->filename;
		scene->filename = scene->filename+".bin";

		fread((void*)(scene->R), sizeof(float), 9, fp);
		fread((void*)(scene->K), sizeof(float), 9, fp);
		fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);  
		fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 


		fread((void*)(&len),    sizeof(unsigned int),   1, fp);
		scene->objects.resize(len);
		if (len>0){
		  totalObjectCount += len;
		  for (int i=0; i<len; ++i){
		      Box3D box;
		      fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
		      fread((void*)(box.base),        sizeof(float), 9, fp);
		      fread((void*)(box.center),      sizeof(float), 3, fp);
		      fread((void*)(box.coeff),       sizeof(float), 3, fp);
		      //process box pad contex oreintation 
		      box = processbox (box, context_pad, grid_size[1]);
		      scene->objects[i]=box;
		      box_id.push_back(i);
		  }
		}
		scenes.push_back(scene);
		totalScenes++;

		if (totalScenes != requestedScene)
		{
			scenes.clear();
			box_id.clear();
			delete scene;
			continue;
		}

		cout << "Scene: " << totalScenes << " Boxes: " << len << " Bin: " << scene->filename << endl << endl;

		//Output files
		FILE* tempname = fopen("temp.txt", "w");
		fprintf(tempname, "%s", s.substr(20).c_str());
		fclose(tempname);
		string tsdffile = "temp.tdsf";

		

		
		float* dataCPUmem = new float[len*3*208*208*100];
		StorageT* dataGPUmem;
		checkCUDA(__LINE__, cudaMalloc(&dataGPUmem, (len)*3*208*208*100*sizeof(float)));
	

		
		compute_TSDF(&scenes, &box_id, dataGPUmem,grid_size,encode_type,scale);
		

		
		checkCUDA(__LINE__, cudaMemcpy(dataCPUmem, dataGPUmem,(len)*3*208*208*100*sizeof(float), cudaMemcpyDeviceToHost) );
	
		

	
		FILE * fid = fopen(tsdffile.c_str(),"wb");
		fwrite(dataCPUmem,sizeof(float),len*3*208*208*100,fid);
		fclose(fid);
		
		

		//clear for workaround
		scenes.clear();
		box_id.clear();

		//free memory
		delete scene;
		delete[] dataCPUmem;
		cudaFree(dataGPUmem);
		
		//Dont calculate others
		break;
	}

	
	fclose(fp);
	return 0;
}


/* USED THIS CODE TO EXTRACT DATA ALREADY
void convertBoxesList()
{
	string box2d = "boxes2d_NYU_po_nb2000.list";
    	
    	FILE* fp2d = fopen(box2d.c_str(),"rb");
    	if (fp2d==NULL) { cout << "Failed to open file: "<< box2d<< endl; exit(EXIT_FAILURE); }


    	while (feof(fp2d)==0) {
      		Scene3D* scene = new Scene3D();
      		unsigned int len = 0;
      		size_t file_size = 0;
      		file_size += fread((void*)(&len), sizeof(unsigned int), 1, fp2d);    
      		if (len==0) break;
      		scene->filename.resize(len);
      		if (len>0) file_size += fread((void*)(scene->filename.data()), sizeof(char), len, fp2d);
     
		int inx = scene->filename.find_last_of("/");
		string output="Boxes//"+scene->filename.substr(inx+1)+".txt";
		FILE* myfile = fopen(output.c_str(), "w");
	

	      	file_size += fread((void*)(scene->R), sizeof(float), 9, fp2d);
		file_size += fread((void*)(scene->K), sizeof(float), 9, fp2d);
		file_size += fread((void*)(&scene->height), sizeof(unsigned int), 1, fp2d);
		file_size += fread((void*)(&scene->width), sizeof(unsigned int), 1, fp2d); 
		file_size += fread((void*)(&len),    sizeof(unsigned int),   1, fp2d);
		scene->objects.resize(len);
      
      		
      		for (int bid = 0;bid<len;bid++){
			//struct Box2D{
			  //unsigned int category;
			  //float tblr[4];
			//};
			Box2D box;
			file_size += fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp2d);
			file_size += fread((void*)(box.tblr),        sizeof(float), 4, fp2d);
			scene->objects_2d_tight.push_back(box);
			 

			fprintf(myfile, "%d %f %f %f %f\n", box.category, box.tblr[0], box.tblr[1], box.tblr[2], box.tblr[3]);


		 	uint8_t hasTarget = 0;
			file_size += fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp2d);
			if (hasTarget>0){ cout<<" sth wrong in line "   << __LINE__ << std::endl; }

			file_size += fread((void*)(box.tblr),   sizeof(float), 4, fp2d);
			scene->objects_2d_full.push_back(box);
			file_size += fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp2d);
			if (hasTarget>0){ cout<<" sth wrong in line "  << __LINE__ << std::endl; }
      		}
		delete scene;
		fclose(myfile);
    	}
    	fclose(fp2d);
}
*/
