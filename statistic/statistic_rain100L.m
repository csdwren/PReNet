clear all

gt_path='../datasets/test/Rain100L/';
jorder_path='../results/Rain100L/Rain100L_JORDER/';
prenet_ssim_path='../results/Rain100L/prenet_ssim/';
prn_ssim_path='../results/Rain100L/prn_ssim/';
prn_recursive_ssim_path='../results/Rain100L/prn_recursive_ssim/';
prenet_recursive_ssim_path='../results/Rain100L/prenet_recursive_ssim/';

struct_model = {
    struct('model_name','prenet_ssim','path',prenet_ssim_path),...
    struct('model_name','prn_ssim','path',prn_ssim_path),...
    struct('model_name','prn_recursive_ssim','path',prn_recursive_ssim_path),...
    struct('model_name','prenet_recursive_ssim','path',prenet_recursive_ssim_path),...
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
            x_true=im2double(imread(fullfile(gt_path,sprintf('norain-%03d.png',iii))));%x_true
            x_true = rgb2ycbcr(x_true);x_true=x_true(:,:,1);
                        
            x = (im2double(imread(fullfile(struct_model{nnn}.path,sprintf('rain-%03d.png',iii)))));
            x = rgb2ycbcr(x);x = x(:,:,1);
            tp = mean(psnr(x,x_true));
            ts = ssim(x*255,x_true*255);
            
            psnrs(iii-nstart,nnn)=tp;
            ssims(iii-nstart,nnn)=ts;
    
        end
    end
    
    fprintf('%s: psnr=%6.4f, ssim=%6.4f\n',struct_model{nnn}.model_name,mean(psnrs(:,nnn)),mean(ssims(:,nnn)));
end

for iii=nstart+1:nstart+nimgs
    for jjj=1:nrain
        x_true=im2double(imread(fullfile(gt_path,sprintf('norain-%03d.png',iii))));%x_true
        x_true = rgb2ycbcr(x_true);
        x_true = x_true(:,:,1);
        
        x = (im2double(imread(fullfile(jorder_path,sprintf('Derained-Rain100L-rain-%03d.png',iii)))));
        x = rgb2ycbcr(x);x = x(:,:,1);
        tp = mean(psnr(x,x_true));
        ts = ssim(x*255,x_true*255);
        
        jorder_psnr(iii-nstart,jjj)=tp;jorder_ssim(iii-nstart,jjj)=ts;
    end
end

fprintf('jorder: psnr=%6.4f, ssim=%6.4f\n',mean(jorder_psnr(:)),mean(jorder_ssim(:)));




