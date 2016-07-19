using Knet
using MAT
RPNWeights=matread("../data/RPN.mat");
gturths=matread("../dataSUNRGBDMeta.mat");
r_props=matread("../data/candidates3d.mat"); 

function main()

batchsize=2;
sumloss = 0;
numloss = 0;

imgs = 1 

for i=1:imgs
	
	#TSDF Computation
	run(`./tsdf $i`);

	tempfilename = open("..//data//temp.txt");
	filename = readall(tempfilename);
	close(tempfilename);
	
	boxnum = prepareScene(filename);


	#3D Input
	TSDFfile=open("../data/temp.tdsf", "r");
	x3D=zeros(Float32, boxnum, 6, 208, 208, 100);
	read!(TSDFfile, x3D);
	close(TSDFfile);
	x3D=permutedims(x3D, [5 4 3 2 1]);
	
	batchcount=floor(boxnum/batchsize);
	if boxnum%batchsize!=0
		batchcount+=1
	end
	

	for j=1:batchcount
		a=1+(j-1)*batchsize;
		b=batchsize*j;
	
		x3d=x3D[:,:,:,:, a:b];
		
		println("burdamiyim");
		y_predict = Network(im3d);

		sumloss += zeroone(y_predict, candidates3d[:, a:b, i]);
		numloss += 1;

		loss = sumloss/numloss;

		println("Scene: $i Batch: $(convert(Int32,j)) Accuracy: $((1-loss)*100)%");
	end
end



@knet function RPNLevel2(x)
	w=par(init=RPNWeights["conv1_w"], dims=(5,5,5,6,96))
	b=par(init=RPNWeights["conv1_b"], dims=(1,1,1,96))
	m=conv(w,x; window=5, padding=2, stride=1)
	m=relu(m.+b)
	m=pool(m; window=2, padding=0, stride=2)

	w=par(init=RPNWeights["conv2_w"], dims=(3,3,3,96,192))
	b=par(init=RPNWeights["conv2_b"], dims=(1,1,1,192))
	m=conv(w,m; window=3, padding=1, stride=1)
	m=relu(m.+b)
	m=pool(m; window=2, stride=2)

	
    w=par(init=RPNWeights["conv3_w"], dims=(3,3,3,192,384))
	b=par(init=RPNWeights["conv3_b"], dims=(1,1,1,384))
	m=conv(w,m; window=3, padding=1, stride=1)
	m=relu(m.+b)
	m=pool(m; window=2, padding=1, stride=1)
	
	
	w=par(init=RPNWeights["conv_cls_score_w"], dims=(5,5,5,384,30))
	b=par(init=RPNWeights["conv_cls_score_b"], dims=(1,1,1,8))
	m=conv(w,m; window=5, padding=2, stride=1)
	m=reshape(m; outdims=(2,2,15,53,53,26))	
	 return wbf(m; out=2, f=:soft)
	 
end	 


function Network(im3d)
    f3d=compile(:RPNLevel2)
	v3d=forw(f3d, im3d)

	predClass=forw(v3d)

	return predClass;
end
	 
	 
	 