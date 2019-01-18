% my 1-20
sizeL=101;
gt_path='E:\derain_dataset\test\ground_truth\';
res_psnrloss_path='C:\Users\Administrator\Desktop\Derain_pytorch\results_res\';
ssimloss_path='C:\Users\Administrator\Desktop\Derain_pytorch\results\';
res_ssimloss_path='C:\Users\Administrator\Desktop\Derain_pytorch\results_res_ssim\';
stcnn_path='C:\Users\Administrator\Desktop\Derain_pytorch\results_STCNN\';

m=5;
nimgs=100;nrain=14;

stncnn_psnr=zeros(nimgs,nrain);
ssimloss_psnr=zeros(nimgs,nrain);
res_psnrloss_psnr=zeros(nimgs,nrain);
res_ssimloss_psnr=zeros(nimgs,nrain);
stncnn_ssim=zeros(nimgs,nrain);
ssimloss_ssim=zeros(nimgs,nrain);
res_psnrloss_ssim=zeros(nimgs,nrain);
res_ssimloss_ssim=zeros(nimgs,nrain);

tp=0;ts=0;te=0;
nstart = 900;
for iii=nstart+1:nstart+nimgs
    for jjj=1:nrain
%         fprintf('img=%d,kernel=%d\n',iii,jjj);
        x_true=im2double(imread([gt_path,num2str(iii),'.jpg']));%x_true
        x_true = rgb2gray(x_true);
        
        x = rgb2gray(im2double(imread([stcnn_path,num2str(iii),'_',num2str(jjj),'.jpg'])));
        tp = psnr(x,x_true);
        ts = ssim(x*255,x_true*255);
        
        stncnn_psnr(iii-nstart,jjj)=tp;stncnn_ssim(iii-nstart,jjj)=ts;
        
        fprintf('stncnn: img=%d kernel=%d: psnr=%6.4f, ssim=%6.4f\n',iii,jjj,tp,ts);
        
        %%
        x = rgb2gray(im2double(imread([res_psnrloss_path,num2str(iii),'_',num2str(jjj),'.jpg'])));
        tp = psnr(x,x_true);
        ts = ssim(x*255,x_true*255);
        
        res_psnrloss_psnr(iii-nstart,jjj)=tp;res_psnrloss_ssim(iii-nstart,jjj)=ts;
        
        fprintf('res_psnrloss: img=%d kernel=%d: psnr=%6.4f, ssim=%6.4f\n',iii,jjj,tp,ts);
        
          %%
        x = rgb2gray(im2double(imread([res_ssimloss_path,num2str(iii),'_',num2str(jjj),'.jpg'])));
        tp = psnr(x,x_true);
        ts = ssim(x*255,x_true*255);
        
        res_ssimloss_psnr(iii-nstart,jjj)=tp;res_ssimloss_ssim(iii-nstart,jjj)=ts;
        
        fprintf('res_ssimloss: img=%d kernel=%d: psnr=%6.4f, ssim=%6.4f\n',iii,jjj,tp,ts);
        
        
          %%
        x = rgb2gray(im2double(imread([ssimloss_path,num2str(iii),'_',num2str(jjj),'.jpg'])));
        tp = psnr(x,x_true);
        ts = ssim(x*255,x_true*255);
        
        ssimloss_psnr(iii-nstart,jjj)=tp;ssimloss_ssim(iii-nstart,jjj)=ts;
        
        fprintf('ssim_loss: img=%d kernel=%d: psnr=%6.4f, ssim=%6.4f\n',iii,jjj,tp,ts);
        

    end
end



fprintf('\nfinal results:\n');
fprintf('stncnn: psnr=%6.4f, ssim=%6.4f\n',mean(stncnn_psnr(:)),mean(stncnn_ssim(:)));
fprintf('ssimloss: psnr=%6.4f, ssim=%6.4f\n',mean(ssimloss_psnr(:)),mean(ssimloss_ssim(:)));
fprintf('res_psnrloss: psnr=%6.4f, ssim=%6.4f\n',mean(res_psnrloss_psnr(:)),mean(res_psnrloss_ssim(:)));
fprintf('res_ssimloss: psnr=%6.4f, ssim=%6.4f\n',mean(res_ssimloss_psnr(:)),mean(res_ssimloss_ssim(:)));




