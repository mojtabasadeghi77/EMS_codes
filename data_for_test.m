clear;clc;
fn3= '/home/mojtabas/US_Sampling/'
fn1='/home/mojtabas/PMW_Global/';
fn2='/home/mojtabas/IR_Global/';

L_Sample=zeros(2,number_sample);
k=1
year=17
years=year+2000
PMW_input=[];
IR_input=[];
L_input=[]
for months=9
    for days=1:5
        days
        load([fn1, 'Date', num2str(year,'%02.f'),num2str(months,'%02.f'),num2str(days,'%02.f'), 'PMW.mat'])
            for tt=1:48
             pp=y(:,:,tt);
             mm(:,:,tt)=pp';
             end
        PMW=mm(251:1250,5626:5625+1750,:); 
        PMW_input=cat(3,PMW_input,PMW);
        load([fn2, 'Date', num2str(year,'%02.f'),num2str(months,'%02.f'),num2str(days,'%02.f'), 'IR.mat'])
            for tt=1:48
            zz=y(:,:,tt);
            rr(:,:,tt)=zz';
            end
         IR=rr(251:1250,5626:5625+1750,:);
         IR_input=cat(3,IR_input,IR);
            t=1;         
    end

 end

save([fn3,'PMW_input5',num2str(year,'%02.f'),num2str(months,'%02.f'),'.mat'],'PMW_input','-v7.3')
save([fn3,'IR_input5',num2str(year,'%02.f'),num2str(months,'%02.f'),'.mat'],'IR_input','-v7.3')
