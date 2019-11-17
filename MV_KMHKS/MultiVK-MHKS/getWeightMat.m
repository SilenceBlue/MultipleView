function FM = getWeightMat(classOne, classTwo)
    lenOne = size(classOne, 1);
    lenTwo = size(classTwo, 1);
    IR = lenTwo/lenOne;
    lenAll = lenOne + lenTwo;
    data = [classOne; classTwo];
    dis_mat = [];
    for i = 1:lenAll
        temp = data(i, :);
        dis = getDis(temp, data);
        dis_mat = [dis_mat, dis];
    end
    %get Radius
    dis_One = dis_mat(1:lenOne,1:lenOne);
    dis_Two = dis_mat(lenOne+1:end,lenOne+1:end);
    disOne = sum(sum(dis_One))/2;
    disTwo = sum(sum(dis_Two))/2;
    avOne = disOne/(lenOne*(lenOne-1));
    avTwo = disTwo/(lenTwo*(lenTwo-1));
    r = avOne + avTwo;
%     r1 = sum(sum(dis_mat, 1), 2)/(lenAll*lenAll);
    F = [];
    max_v = 0;
    min_v = 0;
    iter = 1;
    for i = 1:lenAll
        temp = dis_mat(i, :);
        if i <= lenOne
            ind_r = find(temp <= r);
            n_all = length(ind_r);
            n_min = length(find(ind_r <= lenOne)) - 1;
            n_maj = n_all - n_min;
        end
        if i > lenOne
            ind_r = find(temp <= r);
            n_all = length(ind_r);
            n_min = length(find(ind_r <= lenOne));
            n_maj = n_all - n_min;
        end
        pp = n_min/n_all;
        pn = n_maj/n_all;
        %now we get the number of min samples and maj samples
        %避免P=0时产生NAN
        if pp == 0
            pp = pp+0.00001;
            pn = pn-0.00001;
        end
        if pn == 0
            pn = pn+0.00001;
            pp = pp+0.00001;
        end
        f = -pp*log2(pp) -pn*log2(pn);
        if iter == 1
            max_v = f;
            min_v = f;
            iter = 2;
        end
        if iter == 2
            if f > max_v
                max_v = f;
            end
            if f < min_v
                min_v = f;
            end
        end
        F = [F, f];
    end
    range_v = (max_v - min_v)/10;
    for i = 1:10   
        down_range = min_v + (i - 1)*range_v;
        if i == 10
            up_range = max_v;
            ind_F{i} = find(F >= down_range & F <= up_range);
        else
            up_range = min_v + i*range_v;
            ind_F{i} = find(F >= down_range & F < up_range);
        end       
    end
    for i = 1:10
        F(ind_F{i}) = 1-0.005*(i-1);
    end
    csv = [ones(1,lenOne),ones(1,lenTwo)/IR];
    F = F.*csv;
    FM = diag(F,0);
end

function dis = getDis(x, data)
%
% calculate the distance of x to data
%
    [len, dim] = size(data) ;
    tmpX = repmat(x, len, 1) ;
    dis = sum((data - tmpX).^2,2).^(1/2) ;
end

function par=aveRBFPar(data , size)
    mat_temp = sum(data.^2,2) * ones(1,size) + ones(size,1)*sum(data.^2,2)' - 2* data*data';
    %mat_temp 第一行第二列为第一个样本与第二个样本的距离平方
    mat_temp = sqrt(mat_temp);
    par = (1/size^2) * sum(sum(mat_temp,1),2) ;
    par = real(par);
end