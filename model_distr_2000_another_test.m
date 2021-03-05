clear

% 判断是否属于泊松分布
alpha=0.05;
lambda=9;
output_pred=[];
tagg=1;
col=6;
row=3;
figure(1);
pdf_name='model-distr-another-2000';
% set(gcf,'unit','centimeters','position',[10,10,20,20])
set(gcf,'position',[10,50,750,370]);
% fn=['D:\software\matlab2016\bin\dblp-rels-retag-test-predict-lstm-1992_observe_at_2000_2005_curr.txt'];
for i=0:1:11
%     fn=['D:\dududu\dblp_data\predict\LSTM1\dblp-rels-retag-test-predict-2005_curr_history_',num2str(i),'.txt'];
%     fn_true=['D:\dududu\dblp_data\predict\LSTM1\dblp-rels-retag-test-ture-2005_curr_history_',num2str(i),'.txt'];
    year=2008+i
    fn=['D:\dududu\dblp_data\predict\LSTM_another\dblp-rels-retag-test-predict-',num2str(year),'_curr.txt'];
    fn_true=['D:\dududu\dblp_data\predict\LSTM_another\dblp-rels-retag-test-ture-',num2str(year),'_curr.txt'];
    
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


%     title('Predict the number of publications in 2005','fontname','Times New Roman','Color','black','FontSize',15);
%     subplot(3,4,i+1)
    subplot('Position',[(mod(tagg-1,col))/col+0.04,1-(ceil(tagg/col))/row+0.07,0.75/col,0.75/row])
    set(gca,'box','on','LineWidth',5)
    set(gca,'FontName','Times New Roman','FontSize',10,'LineWidth',1.5);
    set(gca,'FontName','Times New Roman')

%     subplot('Position',[(mod(tagg-1,col))/col+0.03,1-(ceil(tagg/col))/row+0.05,0.75/col,0.75/row])
%     plot(quality_x,quality_pub_pred,'.','MarkerSize',25) 
%     quality_pub_pred=quality_pub_pred(find(quality_pub_pred~=0));
    loglog(quality_pub_pred,'.','MarkerSize',8)
    set(gca,'FontSize',16);
    hold on;
%     plot(quality_x,quality_pub_true,'.','MarkerSize',15) 
%     quality_pub_true=quality_pub_true(find(quality_pub_true~=0));

    loglog(quality_pub_true,'.','MarkerSize',6)
    set(gca,'FontSize',16);
    
    
    sample_size = round(length(pred_data)* .5);
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
% xlabel('the number of publications','FontName','Times New Roman','FontSize',22);
% ylabel('the number of researchers','FontName','Times New Roman','FontSize',22);
% suptitle('Predict the number of publications in 2005')
saveas(gcf,['D:\',pdf_name,'.pdf'])