using Knet
using MAT;



RPNWeights=matread("RPNLast.mat");     #pretrained weights for each layers
RPNgtruths = matread("RPNgtruths.mat"); #RPN ground truth values
RPNgtruths  = RPNgtruths["RPNgt"];



#RPN level 1
@knet function RPN_Level1(x)     # model for receptive field (0.4)^3

	w=par(init=RPNWeights["conv1_w"], dims=(5,5,5,3,96))
	b=par(init=RPNWeights["conv1_b"], dims=(1,1,1,96))
	y=conv(w,x; window=5, padding=1, stride=1)
	y=relu(y.+b)
	y=pool(y; window=2, padding=0, stride=2)

	w=par(init=RPNWeights["conv2_w"], dims=(3,3,3,96,192))
	b=par(init=RPNWeights["conv2_b"], dims=(1,1,1,192))
	y=conv(w,y; window=3, stride=1)
	y=relu(y.+b)
	y=pool(y; window=2, stride=2)

	w=par(init=RPNWeights["reduction1_w"], dims=(1,1,1,192,192))
	b=par(init=RPNWeights["reduction1_w_b"], dims=(1,1,1,192))
	y=relu(y.+b)

	w=par(init=RPNWeights["conv_cls_score1_w"], dims=(2,2,2,192,8))
	b=par(init=RPNWeights["conv_cls_score2_b"], dims=(1,1,1,8))
	
	return soft(y)
end
	


#RPN level 2
@knet function RPN_Level2(x)		# model for receptive field (1.0)^3
	w=par(init=RPNWeights["conv1_w"], dims=(5,5,5,3,96))
	b=par(init=RPNWeights["conv1_b"], dims=(1,1,1,96))
	y=conv(w,x; window=5, padding=1, stride=1)
	y=relu(y.+b)
	y=pool(y; window=2, padding=0, stride=2)

	w=par(init=RPNWeights["conv2_w"], dims=(3,3,3,96,192))
	b=par(init=RPNWeights["conv2_b"], dims=(1,1,1,192))
	y=conv(w,y; window=3, stride=1)
	y=relu(y.+b)
	y=pool(y; window=2, stride=2)

	w=par(init=RPNWeights["conv3_w"], dims=(3,3,3,192,384))
	b=par(init=RPNWeights["conv3_b"], dims=(1,1,1,384))
	y=conv(w,y; window=3, stride=1)
	y=relu(y.+b)
    y=pool(y; window=2, stride=2)
	
	
	w=par(init=RPNWeights["conv_cls_score_w"], dims=(5,5,5,384,30))
	b=par(init=RPNWeights["conv_cls_score_b"], dims=(1,1,1,30))
	y=conv(w,y; window=5, stride=1)
	
	
	return soft(y)
end
	


batchsize=2;		#minibatch size 
loss = 0;
count = 0;

scene_num = 1     # just 1 test file is added to github to save the space, rest can be found at: http://dss.cs.princeton.edu/Release/sunrgbd_dss_data/SUNRGBD/kv2/kinect2data/


for i=1:scene_num

	run(`./tsdf $i`);


	 num_box = 2000;    #box number 
	

	TSDFfile=open("temp.tdsf", "r");									
	
	box_3D=zeros(Float32, num_box , 3, 30, 30, 30);
	
	read!(TSDFfile, box_3D);				#reading the tsdf file which contains the computed tsdf value of 3D image
	close(TSDFfile);
	
	
	batchcount=num_box /batchsize;
	
	if num_box %batchsize!=0
		batchcount+=1
	end
	
	
	for j=1:batchcount
		b1=1+(j-1)*batchsize;
		b2=batchsize*j;

		b3d=box_3D[:,:,:,:, b1:b2];
		
		println("im here");
		
		RPN3d=compile(:RPN_Level1)				#we select level1 or level2 to run as desired receptive field
		#RPN3d2=compile(:RPN_Level2)

		print(size(b3d))

	 
		pred=forw(RPN3d, b3d)			#forward operation (I do not train the model, use pre-trained weights)
        #pred=forw(RPN3d2, b3d)
		
		println(pred);

		loss += zeroone(pred, RPNgtruths[:, b1:b2, i]);
		count += 1;

		loss = loss/count;

		@printf("3D image: %d Box:%d  tst Accuracy: %g\n", i, j, ((1-loss)*100%))  # comparing the bounding boxes with ground truth values
		
	end
	
end

