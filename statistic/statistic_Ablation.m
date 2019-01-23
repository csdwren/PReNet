
clear all;
close all;

gt_path='../datasets/test/Rain100H/';

PReNet = '../results/Ablation/PReNet/';
PReNet6_x = '../results/Ablation/PReNet6_x/';
PReNet5_LSTM = '../results/Ablation/PReNet5_LSTM/';
PReNet6_LSTM = '../results/Ablation/PReNet6_LSTM/';
PReNet7_LSTM = '../results/Ablation/PReNet7_LSTM/';
PReNet6_GRU = '../results/Ablation/PReNet6_GRU/';
PReNet_RecSSIM = '../results/Ablation/PReNet_RecSSIM/';

 
struct_model = {
           struct('model_name','PReNet5_LSTM','path',PReNet5_LSTM),...
           struct('model_name','PReNet6_LSTM','path',PReNet6_LSTM),...
           struct('model_name','PReNet7_LSTM','path',PReNet7_LSTM),...
           struct('model_name','PReNet6_GRU','path',PReNet6_GRU),...
           struct('model_name','PReNet','path',PReNet),...
           struct('model_name','PReNet_RecSSIM','path',PReNet_RecSSIM),...
           struct('model_name','PReNet6_x','path',PReNet6_x),...
    };


nimgs=100;nrain=1;
nmodel = length(struct_model);

psnrs = zeros(nimgs,nmodel);
ssims = psnrs;

for nnn = 1:nmodel
    
    tp=0;ts=0;te=0;
    nstart = 0;
    for iii=nstart+1:nstart+nimgs
        for jjj=1:nrain
            %         fprintf('img=%d,kernel=%d\n',iii,jjj);
            x_true=im2double(imread(fullfile(gt_path,sprintf('norain-%03d.png',iii))));%x_true
            x_true = rgb2ycbcr(x_true);x_true=x_true(:,:,1);
            

            %%
            x = (im2double(imread(fullfile(struct_model{nnn}.path,sprintf('rain-%03d.png',iii)))));
            x = rgb2ycbcr(x);x = x(:,:,1);
            tp = mean(psnr(x,x_true));
            ts = ssim(x*255,x_true*255);
            
            psnrs(iii-nstart,nnn)=tp;
            ssims(iii-nstart,nnn)=ts;
            
            %
        end
    end
    
    fprintf('%s: psnr=%6.4f, ssim=%6.4f\n',struct_model{nnn}.model_name,mean(psnrs(:,nnn)),mean(ssims(:,nnn)));
    
end

