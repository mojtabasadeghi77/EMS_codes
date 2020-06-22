figure(1)
subplot(4,4,1)
i=randi(4650)
imagesc(IR_Sample(:,:,i))
colorbar
title('IR')
hcb2=colorbar;
title(hcb2, 'C','FontWeight','bold')
subplot(4,4,2)
l=185
imagesc(PMW_Sample(:,:,i))
title('PMW')
hcb2=colorbar;
title(hcb2, 'mm/hr','FontWeight','bold')
subplot(4,4,3)
l=185
imagesc(Loc_Sample(:,:,1,i))
title('Lat')
colorbar
subplot(4,4,4)
l=185
imagesc(Loc_Sample(:,:,2,i))
title('Lon')
colorbar
%%
subplot(4,4,5)
i=randi(4650)
imagesc(IR_Sample(:,:,i))
colorbar

hcb2=colorbar;
title(hcb2, 'C','FontWeight','bold')
subplot(4,4,6)
l=185
imagesc(PMW_Sample(:,:,i))

hcb2=colorbar;
title(hcb2, 'mm/hr','FontWeight','bold')
subplot(4,4,7)
l=185
imagesc(Loc_Sample(:,:,1,i))

colorbar
subplot(4,4,8)
l=185
imagesc(Loc_Sample(:,:,2,i))

colorbar
%%
subplot(4,4,9)
i=randi(4650)
imagesc(IR_Sample(:,:,i))
colorbar

hcb2=colorbar;
title(hcb2, 'C','FontWeight','bold')
subplot(4,4,10)
l=185
imagesc(PMW_Sample(:,:,i))

hcb2=colorbar;
title(hcb2, 'mm/hr','FontWeight','bold')
subplot(4,4,11)
l=185
imagesc(Loc_Sample(:,:,1,i))

colorbar
subplot(4,4,12)
l=185
imagesc(Loc_Sample(:,:,2,i))

colorbar
%%
subplot(4,4,13)
i=randi(4650)
imagesc(IR_Sample(:,:,i))
colorbar

hcb2=colorbar;
title(hcb2, 'C','FontWeight','bold')
subplot(4,4,14)
l=185
imagesc(PMW_Sample(:,:,i))

hcb2=colorbar;
title(hcb2, 'mm/hr','FontWeight','bold')
subplot(4,4,15)
l=185
imagesc(Loc_Sample(:,:,1,i))

colorbar
subplot(4,4,16)
l=185
imagesc(Loc_Sample(:,:,2,i))

colorbar