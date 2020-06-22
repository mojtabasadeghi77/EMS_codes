%%  format is binary 2-byte short int  (big-endian)
clc;
clear all;
cd '/zfs_data3/goesdata/thunder/ir_netcdf/globir_04/2017/'
fn1='/home/mojtabas/IR_Global/'
%fn2='/home/mojtabas/IR_Junk/'
fn2='/home/mojtabas/IR_JUNK1/'

for year =17
    years=year+2000
    for months=12
        months
        for days = 1:eomday(years,months);
            %
            days
            y=[];
            for hr=0:23
                for min=15:30:45
             gunzip(['bglob', num2str(year,'%02.f'), num2str(months,'%02.f'),num2str(days,'%02.f'),num2str(hr,'%02.f'),num2str(min,'%02.f'),'.bin.gz'],fn2)
             t = fopen([fn2,'bglob', num2str(year,'%02.f'), num2str(months,'%02.f'),num2str(days,'%02.f'),num2str(hr,'%02.f'),num2str(min,'%02.f'),'.bin'],'r','b');
             x=fread(t, [9000, 3000], 'short')./100;
             x(x<0)=NaN;
             y=cat(3,y,x);
                end
            end
             save([fn1, 'Date', num2str(year,'%02.f'),num2str(months,'%02.f'),num2str(days,'%02.f'), 'IR.mat'], 'y','-v7.3'); 
        which_dir = fn2;
        dinfo = dir(which_dir);
        dinfo([dinfo.isdir]) = [];   %skip directories
        filenames = fullfile(which_dir, {dinfo.name});
        delete( filenames{:} )
        end 
    end   
end
display('finished')
