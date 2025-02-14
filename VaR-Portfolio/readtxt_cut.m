fid=fopen('data_5_100_300_5.txt','r');
fout=fopen('new.txt','w');
i=0;
while ~feof(fid) 
    tline=fgetl(fid);
    if i == 0
        i=i+1;
        continue
    else
         fprintf(fout,'%s\n',tline);
    end
    i=i+1;
end
data = importdata('new.txt');