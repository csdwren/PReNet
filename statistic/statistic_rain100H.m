
clear all;
close all;

gt_path='../datasets/test/Rain100H/';
JORDER_path='../results/Rain100H/Rain100H_JORDER/';
RESCAN_path='../results/Rain100H/Rain100H_RESCAN/';

PReNet = '../results/Rain100H/PReNet/';
PReNet_r = '../results/Rain100H/PReNet_r/';
PRN = '../results/Rain100H/PRN6/';
PRN_r = '../results/Rain100H/PRN_r/';
 
struct_model = {
          struct('model_name','PReNet','path',PReNet),...
          struct('model_name','PReNet_r','path',PReNet_r),...
          struct('model_name','PRN','path',PRN),...
          struct('model_name','PRN_r','path',PRN_r),...
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
        
        x = (im2double(imread(fullfile(JORDER_path,sprintf('Derained-Rain100H-rain-%03d.png',iii)))));
        x = rgb2ycbcr(x);x = x(:,:,1);
        tp = mean(psnr(x,x_true));
        ts = ssim(x*255,x_true*255);
        
        jorder_psnr(iii-nstart,jjj)=tp;jorder_ssim(iii-nstart,jjj)=ts;
        
        %         fprintf('pku: img=%d: psnr=%6.4f, ssim=%6.4f\n',iii,tp,ts);
    end
end

fprintf('JORDER: psnr=%6.4f, ssim=%6.4f\n',mean(jorder_psnr(:)),mean(jorder_ssim(:)));



%%
for iii=nstart+1:nstart+nimgs
    for jjj=1:nrain
        %         fprintf('img=%d,kernel=%d\n',iii,jjj);
        x_true=im2double(imread(fullfile(gt_path,sprintf('norain-%03d.png',iii))));%x_true
        x_true = rgb2ycbcr(x_true);
        x_true = x_true(:,:,1);
        
        x = (im2double(imread(fullfile(RESCAN_path,sprintf('%d_3.png',iii)))));
        x = rgb2ycbcr(x);x = x(:,:,1);
        tp = mean(psnr(x,x_true));
        ts = ssim(x*255,x_true*255);
        
        rescan_psnr(iii-nstart,jjj)=tp;rescan_ssim(iii-nstart,jjj)=ts;
        
        %         fprintf('pku: img=%d: psnr=%6.4f, ssim=%6.4f\n',iii,tp,ts);
    end
end

fprintf('RESCAN: psnr=%6.4f, ssim=%6.4f\n',mean(rescan_psnr(:)),mean(rescan_ssim(:)));




