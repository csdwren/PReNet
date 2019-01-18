clear all

gt_path = '../datasets/test/test12/groundtruth/';

all_lstm_ssim_path='../results/test12/results_all_lstm_ssim/';
lstm_ssim_path='../results/test12/results_light_lstm_ssim/';
resnet_ssim_path='../results/test12/results_new_light_resnet_ssim/';
resnet_recursive_ssim_path='../results/test12/results_new_light_resnet_recursive_ssim/';
lstm_recursive_ssim_path='../results/test12/results_new_light_lstm_recursive_ssim/';

struct_model = {
    struct('model_name','all_lstm_ssim','path',all_lstm_ssim_path),...
    struct('model_name','lstm_ssim','path',lstm_ssim_path),...
    struct('model_name','resnet_ssim','path',resnet_ssim_path),...
    struct('model_name','resnet_recursive_ssim','path',resnet_recursive_ssim_path),...
    struct('model_name','lstm_recursive_ssim','path',lstm_recursive_ssim_path),...
    };


nimgs=12;nrain=1;
nmodel = length(struct_model);

psnrs = zeros(nimgs,nmodel);
ssims = psnrs;

for nnn = 1:nmodel
    
    tp=0;ts=0;te=0;
    nstart = 0;
    for iii=nstart+1:nstart+nimgs
        for jjj=1:nrain
            %         fprintf('img=%d,kernel=%d\n',iii,jjj);
            x_true=im2double(imread(fullfile(gt_path,sprintf('%d.png',iii))));%x_true
            x_true = rgb2ycbcr(x_true);x_true=x_true(:,:,1);
            
            
            %%
            x = (im2double(imread(fullfile(struct_model{nnn}.path,sprintf('%d.png',iii)))));
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



