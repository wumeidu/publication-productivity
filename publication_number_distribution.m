clear

alpha=0.05;
lambda=9;
output_pred=[];
tagg=1;
col=6;
row=3;
figure(1);
pdf_name='publication-number-distribution';

set(gcf,'position',[10,50,750,370]);

for i=0:1:17

    year=2001+i
    fn=['D:\dblp-rels-retag-test-predict-',num2str(year),'_curr_history_15.txt'];
    fn_true=['D:\dblp-rels-retag-test-ture-',num2str(year),'_curr_history_15.txt'];
    
    fid=importdata(fn);
    fid_true=importdata(fn_true);

    pred_data=round(fid');
    true_data=fid_true;

    quality_x=1:1:100;
    quality_pub_pred=[];
    quality_pub_true=[];
    for num=1:length(quality_x)
        quality_pub_pred(num)=length(find(pred_data==quality_x(num)));
        quality_pub_true(num)=length(find(true_data==quality_x(num)));
    end


    subplot('Position',[(mod(tagg-1,col))/col+0.04,1-(ceil(tagg/col))/row+0.07,0.75/col,0.75/row])
    set(gca,'box','on','LineWidth',5)
    set(gca,'FontName','Times New Roman','FontSize',10,'LineWidth',1.5);
    set(gca,'FontName','Times New Roman')


    loglog(quality_pub_pred,'.','MarkerSize',8)
    set(gca,'FontSize',16);
    hold on;


    loglog(quality_pub_true,'.','MarkerSize',6)
    set(gca,'FontSize',16);
    
    
    sample_size = round(length(pred_data)* .2);
    [h,p] = kstest2(pred_data(1:sample_size), true_data(1:sample_size), 'Alpha',0.05  );
    s4=['y=',num2str(year)];
    text((mod(tagg-1,col))/col+1.25, (1-(ceil(1/col))/row+30-18)*45/40    , s4 ,'FontSize', 7, 'Fontname', 'Times New Roman' )
    s3=['p=',num2str(roundn(p,-2))];
    text((mod(tagg-1,col))/col+1.25, (1-(ceil(1/col))/row+30-26)*45/40    , s3 ,'FontSize', 7, 'Fontname', 'Times New Roman' )
    axis([0.99,150,0.9,5000]);
    axis on;
    set(gca,'XtickMode','manual','XTick',[1,100]);
    set(gca,'YtickMode','manual','YTick',[1,100,1000]);
    set(gca,'FontName','Times New Roman','FontSize',7);
    hold on
    tagg=tagg+1;
end

saveas(gcf,['D:\',pdf_name,'.pdf'])
