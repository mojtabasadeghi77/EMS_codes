
clear;clc;
fn3= '/home/mojtabas/US_Sampling/'
fn1='/home/mojtabas/PMW_Global/';
fn2='/home/mojtabas/IR_Global/';
sample_size=128;
number_sample=150;
PMW_Sample=zeros(128,128,number_sample);
IR_Sample=zeros(128,128,number_sample);
L_Sample=zeros(2,number_sample);
k=1
year=17
years=year+2000
for months=7
    for days=1:eomday(years,months)
        days
        load([fn1, 'Date', num2str(year,'%02.f'),num2str(months,'%02.f'),num2str(days,'%02.f'), 'PMW.mat'])
            for tt=1:48
             pp=y(:,:,tt);
             mm(:,:,tt)=pp';
             end
        PMW=mm(251:1250,5626:5625+1750,:); 
        load([fn2, 'Date', num2str(year,'%02.f'),num2str(months,'%02.f'),num2str(days,'%02.f'), 'IR.mat'])
            for tt=1:48
            zz=y(:,:,tt);
            rr(:,:,tt)=zz';
            end
        IR=rr(251:1250,5626:5625+1750,:);
            t=1;
            while t <number_sample+1
                r1= randi(size(PMW,1)-sample_size-1);
                r2= randi(size(PMW,2)-sample_size-1);
                r3=randi([1 48]);
                PMW_mat=PMW(r1:r1+sample_size-1,r2:r2+sample_size-1,r3);
                 if isnan(mean(PMW_mat(:)))
                 else
                 num=sum(sum(PMW_mat(:,:)~=0));
                 ratio=num/(sample_size).^2;
                    if ratio>0.25 & nanmean(PMW_mat(:))>1
                        l=IR(r1:r1+sample_size-1,r2:r2+sample_size-1,r3);
                        if isnan(mean(l(:)))
                        else
                            PMW_Sample(:,:,k)=PMW_mat;
                            IR_Sample(:,:,k)=l;
                            L_Sample(1,k)=r1;
                            L_Sample(2,k)=r2;
                            t=t+1;
                            k=k+1;
                        end
                    end
                 end
            end
        end
 end
save([fn3,'PMW_Sample',num2str(year,'%02.f'),num2str(months,'%02.f'),'.mat'],'PMW_Sample','-v7.3')
save([fn3,'IR_Sample',num2str(year,'%02.f'),num2str(months,'%02.f'),'.mat'],'IR_Sample','-v7.3')
