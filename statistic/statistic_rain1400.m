
clear all;
close all;

gt_path='../datasets/test/Rain1400/ground_truth/';

PReNet = '../results/Rain1400/PReNet/';
PReNet_r = '../results/Rain1400/PReNet_r/';
PRN = '../results/Rain1400/PRN6/';
PRN_r = '../results/Rain1400/PRN_r/';
 
struct_model = {
          struct('model_name','PReNet','path',PReNet),...
          struct('model_name','PReNet_r','path',PReNet_r),...
          struct('model_name','PRN','path',PRN),...
          struct('model_name','PRN_r','path',PRN_r),...
    };


nimgs=100;nrain=14;
nmodel = length(struct_model);


for nnn = 1:nmodel
    
    tp=0;ts=0;te=0;
    nstart = 900;
    count = 1;
    for iii=nstart+1:nstart+nimgs
        for jjj=1:nrain
            %         fprintf('img=%d,kernel=%d\n',iii,jjj);
            x_true=im2double(imread(fullfile(gt_path,sprintf('%d.jpg',iii))));%x_true
            x_true = rgb2ycbcr(x_true);x_true=x_true(:,:,1);
            

            %%
            x = (im2double(imread(fullfile(struct_model{nnn}.path,sprintf('%d_%d.jpg',iii,jjj)))));
            x = rgb2ycbcr(x);x = x(:,:,1);
            tp = mean(psnr(x,x_true));
            ts = ssim(x*255,x_true*255);
            
            psnrs(count,nnn)=tp;
            ssims(count,nnn)=ts;
            
            count = count + 1;
            
            %
        end
    end
    
    fprintf('%s: psnr=%6.4f, ssim=%6.4f\n',struct_model{nnn}.model_name,mean(psnrs(:,nnn)),mean(ssims(:,nnn)));
    
end


