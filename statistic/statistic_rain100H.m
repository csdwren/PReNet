
clear all;

gt_path='../datasets/test/Rain100H/';
jorder_path='../results/Rain100H/Rain100H_JORDER/';
new_lstm_mse_path='../results/Rain100H/results_new_lstm_mse/';
new_resnet_ssim_path='../results/Rain100H/results_new_resnet_ssim/';

lstm_ssim_multi0_path='../results/Rain100H/results_lstm_ssim_multiloss/s0/';
lstm_ssim_multi1_path='../results/Rain100H/results_lstm_ssim_multiloss/s1/';
lstm_ssim_multi2_path='../results/Rain100H/results_lstm_ssim_multiloss/s2/';
lstm_ssim_multi3_path='../results/Rain100H/results_lstm_ssim_multiloss/s3/';

lstm_ssim_path='../results/Rain100H/results_new_lstm_ssim/';
onelstm_ssim_path='../results/Rain100H/results_new_1lstm_ssim/';
mask1lstm_ssim_path='../results/Rain100H/results_new_mask1lstm_ssim/';

recursive1resnet_ssim_path='../results/Rain100H/results_new_recursive1resnet_ssim/';
recursive1lstm_ssim_path='../results/Rain100H/results_new_recursive1lstm_ssim/';


struct_model = {
    struct('model_name','lstm_multi0_ssim','path',lstm_ssim_multi0_path),...
    struct('model_name','lstm_multi1_ssim','path',lstm_ssim_multi1_path),...
    struct('model_name','lstm_multi2_ssim','path',lstm_ssim_multi2_path),...
    struct('model_name','lstm_multi3_ssim','path',lstm_ssim_multi3_path),...
    struct('model_name','lstm_ssim','path',lstm_ssim_path),...
    struct('model_name','1lstm_ssim','path',onelstm_ssim_path),...
    struct('model_name','mask1lstm_ssim','path',mask1lstm_ssim_path),...
    struct('model_name','recursive1lstm_ssim','path',recursive1lstm_ssim_path),...
    struct('model_name','recursive1resnet_ssim','path',recursive1resnet_ssim_path),...
%     struct('model_name','lstm_s5_ssim_path','path',lstm_s5_ssim_path),...
%     struct('model_name','lstm_s2_ssim_path','path',lstm_s2_ssim_path),...
%     struct('model_name','lstm_mse2_path','path',lstm_mse2_path),...
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

for iii=nstart+1:nstart+nimgs
    for jjj=1:nrain
        %         fprintf('img=%d,kernel=%d\n',iii,jjj);
        x_true=im2double(imread(fullfile(gt_path,sprintf('norain-%03d.png',iii))));%x_true
        x_true = rgb2ycbcr(x_true);
        x_true = x_true(:,:,1);
        
        x = (im2double(imread(fullfile(jorder_path,sprintf('Derained-Rain100H-rain-%03d.png',iii)))));
        x = rgb2ycbcr(x);x = x(:,:,1);
        tp = mean(psnr(x,x_true));
        ts = ssim(x*255,x_true*255);
        
        jorder_psnr(iii-nstart,jjj)=tp;jorder_ssim(iii-nstart,jjj)=ts;
        
    end
end

fprintf('jorder: psnr=%6.4f, ssim=%6.4f\n',mean(jorder_psnr(:)),mean(jorder_ssim(:)));




