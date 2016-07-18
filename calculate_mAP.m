
myFolder = '\\BUKET-PC\Users\Buket\Desktop\NYU proposals'; 
filePattern = fullfile(myFolder, 'NYU000*.mat');
matFiles = dir(filePattern);
for k = 1:length(matFiles)
	matFilename = fullfile(myFolder, matFiles(k).name)
	matData = load(matFilename); 
	
	hasField = isfield(matData, 'candidates3d');

for row = 1 : size(candidates3d,1)
		iou= {candidates3d(1:length(candidates3d)).iou};
        
 count=0;
 for i=1:length(candidates3d)
     if iou{i}>0.25
         count=count+1;
     end   
 end
end
      mAP= count/ length(candidates3d)
    end
