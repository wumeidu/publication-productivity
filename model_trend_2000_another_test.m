clear
tagg=1
predict_thresd=15

fn =['D:\dududu\dblp_data\coauthor_evolu_data\dblp-rels-retag-1995-2019_1951_hyper_degree_squence_start_at_2000.txt'];
fid =importdata(fn); 
   
hyperdegrees_null_1= fid;
hyperdegrees_null=sum(hyperdegrees_null_1(:,1:13),2);    % sum(a,dim):dim=1/2 ����/����� 9�ĳ�13
hyperdegrees_null=hyperdegrees_null(find(hyperdegrees_null<=predict_thresd));

intial_hyperdegree_value=[1:1:predict_thresd];

fid=fid(find(sum(hyperdegrees_null_1(:,1:13),2) <=predict_thresd),:);
predict_hyperdegrees=sum(fid(:,1:13),2   );
nodes_set_hyperdegree=predict_hyperdegrees;


end_y=[2008:1:2019];
beta=-.00000
gamma=20
col=6
row=3
 %13 %����ҪԤ��Ľڵ�
% s2='trend_RNN_20010_2018'

s2='model-trend-another-2000'

%����hazard���� 
set(gcf,'unit','normalized','position',[0.1,0.1,0.37 ,0.37])
for ss=2008:2019%7:15%
    fn=['D:\dududu\dblp_data\predict\LSTM_another\dblp-rels-retag-test-predict-', num2str(ss),'_curr.txt'];
    fn_1=['D:\dududu\dblp_data\predict\LSTM_another\dblp-rels-retag-test-ture-', num2str(ss),'_curr.txt'];
%     
%    fn=['E:\gaoshu1_2016\dblp-rels-retag-test-predict-ac-', num2str(ss),'_curr.txt'];
%    fn_1=['E:\gaoshu1_2016\dblp-rels-retag-test-ture-ac-', num2str(ss),'_curr.txt'];
    
%      fn=['E:\dl_productivity_predict\dblp-rels-retag-test-predict-', num2str(ss),'_curr_aa.txt'];
%      fn_1=['E:\dl_productivity_predict\dblp-rels-retag-test-ture-', num2str(ss),'_curr_aa.txt'];    
    
    fid =importdata(fn); % 5869
    fid_1 =importdata(fn_1);
    
    fid =fid(:,1);  % 5869
    fid_1 =fid_1(:,1); % 5869
   
    predict_hyperdegrees= round(fid);   % ȡ�� % 5869
    nodes_set_hyperdegree= round(fid_1);% 5869

     
    average_hyperdegree=0;

    for ii =1:length(intial_hyperdegree_value) 
        average_hyperdegree(ii)=    mean(nodes_set_hyperdegree (find(hyperdegrees_null== intial_hyperdegree_value(ii) )));

    end

  
    predict_average_hyperdegree=0;
    for ii =1:length(intial_hyperdegree_value) 
        predict_average_hyperdegree(ii)=    mean(predict_hyperdegrees (find(hyperdegrees_null== intial_hyperdegree_value(ii) )));

    end

   

    subplot('Position',[(mod(tagg-1,col))/col+0.03,1-(ceil(tagg/col))/row+0.05,0.75/col,0.75/row])
    set(gca,'box','on','LineWidth',0.6)
    set(gca,'FontName','Times New Roman')
    axis([1 predict_thresd+1     0 35]);
    hold on
    set(gca,'XTick',[1,predict_thresd])
    hold on
    set(gca,'YTick',[1, 20])
    hold on
    
    hold on
    %subplot(1, length(end_y), ss)
    plot(intial_hyperdegree_value,  predict_average_hyperdegree , '-b','LineWidth', 0.9 ,'markersize',0.8,'Color',[51/255,102/255,255/255] )
    hold on
    plot(intial_hyperdegree_value,    average_hyperdegree, 'o','LineWidth',0.001,'markersize',1.8 , 'MarkerFaceColor',[0.85,0.33,0.01], 'MarkeredgeColor',[0.85,0.33,0.01])
    hold on
   
    [rho,pval]= corr(predict_hyperdegrees ,nodes_set_hyperdegree,'Type', 'Pearson');
    %plot(intial_hyperdegree_value,    intial_hyperdegree_value,   'g')
    ss2=ss-2007;%;
    s3=['y=',num2str(end_y(ss2)) ];
    text((mod(tagg-1,col))/col+2,( 1-(ceil(1/col))/row+30-11 )*45/30   , s3 ,'FontSize', 8, 'Fontname', 'Times New Roman' )
    s3=['s_1=',num2str(round(rho,3))];
    text((mod(tagg-1,col))/col+2, (1-(ceil(1/col))/row+30-14)*45/30    , s3 ,'FontSize', 8, 'Fontname', 'Times New Roman' )
    
    predict_hyperdegrees_1=sort(predict_hyperdegrees);
    nodes_set_hyperdegree_1=sort(nodes_set_hyperdegree);
    
    
    [rho,pval]= corr(predict_hyperdegrees_1 ,nodes_set_hyperdegree_1,'Type', 'Pearson');
    s3=['s_2=',num2str(round(rho,3)) ];
    text((mod(tagg-1,col))/col+2, (1-(ceil(1/col))/row+30-17)*45/30 , s3 ,'FontSize', 8, 'Fontname', 'Times New Roman' )
    
    hold off
    tagg=tagg+1
    
end

saveas(gcf,['D:\',s2,'.pdf'])