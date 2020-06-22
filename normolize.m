PMW=[];
IR=[];
LC=[];

%%
PMW=cat(3,PMW,PMW_Sample);
IR=cat(3,IR,IR_Sample);
LC=cat(2,LC,L_Sample);
%%
save('PMW.mat','PMW','-v7.3')
save('IR.mat','IR','-v7.3')
save('LC.mat','LC','-v7.3')
%%
%fn='C:\Mojtaba\CNN\Global_PMW\Date_Read\Sampling\'
number_sample=28950;
x=reshape(PMW,[128*128*number_sample,1,1]);
min_x=nanmin(x)
std_x=nanstd(x)
PMW_norm=(PMW-min_x)/std_x;
PMW_norm(isnan(PMW_norm))=-0.01;
%%
save('PMW_norm.mat','PMW_norm','-v7.3')
%%
x=reshape(IR,[128*128*number_sample,1,1]);
min_x=nanmin(x)
std_x=nanstd(x)
IR_norm=(IR-min_x)/std_x;
IR_norm(isnan(IR_norm))=-0.01;
%%
save('IR_norm.mat','IR_norm','-v7.3')
%%
for k= 1:number_sample
    for i=1:128
         for j=1:128
           sample(i,j,1,k)=LC(1,k)+i-1;
           sample(i,j,2,k)=LC(2,k)+j-1;
         end
    end
end
x=reshape(sample(:,:,1,:),[1000*1750*number_sample,1,1]);
y=reshape(sample(:,:,2,:),[1000*1750*number_sample,1,1]);
min_x=nanmin(x)
std_x=nanstd(x)
min_y=nanmin(y)
std_y=nanstd(y)
LC_norm(:,:,1,:)=(sample(:,:,1,:)-min_x)/std_x;
LC_norm(:,:,2,:)=(sample(:,:,2,:)-min_y)/std_y;
LC_norm(isnan(LC_norm))=-0.01;
save('LC_norm.mat','LC_norm','-v7.3')