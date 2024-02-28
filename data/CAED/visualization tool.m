function show_annotations(imdir, anno, dataset)
% draw images with annotation, 
% imdir : image directory
% anno : annotation 
% dataset : 0 for collective activity dataset (2009)
%           1 for new activity dataset (2012 eccv)

imfiles = dir(fullfile(imdir, '*.jpg'));
[vidanno] = get_video_annotations(anno, dataset);
figure(1);
cmap = colormap;
for i = 1:3:length(imfiles)
    im = imread(fullfile(imdir, imfiles(i).name));
    imshow(im);
    
    for j = 1:length(vidanno(i).tid)
        idx = mod(vidanno(i).tid(j) * 10, 64) + 1;
        col = cmap(idx, :);
        % draw bboxes
        rectangle('position', vidanno(i).bbox(:, j), 'linewidth', 2, 'edgecolor', col);
        % show atomic properties
        text(max(1, vidanno(i).bbox(1, j) + 5), max(1, vidanno(i).bbox(2, j) + vidanno(i).bbox(4, j) - 15), [vidanno(i).actions{j} vidanno(i).poses{j}], ...
                    'LineWidth', 1, 'EdgeColor', 'k', 'BackgroundColor', 'w', 'Color', 'k', 'FontSize', 25);
    end
    % show interactions
    for j = 1:length(vidanno(i).interactions)
        pt = [(vidanno(i).ilines(1, j) + vidanno(i).ilines(3, j)) / 2, (vidanno(i).ilines(2, j) + vidanno(i).ilines(4, j)) / 2];
        
        line([vidanno(i).ilines(1, j), vidanno(i).ilines(3, j)], [vidanno(i).ilines(2, j), vidanno(i).ilines(4, j)], ...
                'color', 'w', 'linestyle', '--', 'linewidth', 3);
            
        text(pt(1), pt(2), vidanno(i).interactions{j}, ...
                    'LineWidth', 1, 'EdgeColor', 'k', 'BackgroundColor', 'w', 'Color', 'k', 'FontSize', 25);
    end
    % 
    text(size(im, 2) / 2, 30, vidanno(i).clabel, 'HorizontalAlignment', 'center', ...
                    'LineWidth', 3, 'EdgeColor', 'k', 'BackgroundColor', 'w', 'Color', 'k', 'FontSize', 45);
                
    drawnow;
end

end

function [vidanno] = get_video_annotations(anno, dataset)

annotation = struct('clabel', cell(anno.nframe, 1), ... 
                    'ilabels', cell(anno.nframe, 1), ...
                    'i_tid', cell(anno.nframe, 1), ...
                    'act_labels', cell(anno.nframe, 1), ...
                    'pose_labels', cell(anno.nframe, 1), ...
                    'a_tid', cell(anno.nframe, 1));

if(dataset == 0)
    cstr = {'NA' 'Crossing' 'Waiting' 'Queuing' 'Walking' 'Talking'};
    astr = {'W' 'S'};
    istr = {'NA' 'AP' 'LV' 'PB' 'FE' 'WS' 'SQ' 'SS'};
else
    cstr = {'NA' 'Gathering' 'Talking' 'Dismissal' 'Walking together' 'Chasing' 'Queuing'};
    astr = {'S' 'W' 'R'};
    istr = {'NA' 'AP' 'WO' 'WS' 'WR' 'RS' 'RR' 'FE' 'SR'};
end
pstr = {'R' 'RF' 'F' 'LF' 'L' 'LB' 'B' 'RB'};


vidanno = struct('clabel', cell(1, anno.nframe), ...
                 'tid', [], 'bbox', [], 'actions', [], 'poses', [], ...
                 'interactions', [], 'ilines', []);
             
for j = 1:anno.nframe
    if(dataset == 0)
        idx = ceil(j / 10) * 10 + 1;
        if(idx > anno.nframe)
            idx = anno.nframe;
        end
        clabel = anno.collective(idx);
    else
        clabel = anno.collective(j);
    end
    vidanno(j).clabel = cstr(clabel);
end

for i = 1:length(anno.people)
    for j = 1:length(anno.people(i).time)
        fr = anno.people(i).time(j);
        
        vidanno(fr).tid(end+1) = i;
        vidanno(fr).bbox(:, end+1) = anno.people(i).sbbs(:, j);
        
        if(anno.people(i).attr(1, j) <= 0)
            vidanno(fr).poses{end+1} = 'NA';
        else
            vidanno(fr).poses{end+1} = pstr(anno.people(i).attr(1, j));
        end
        if(anno.people(i).attr(2, j) <= 0)
            vidanno(fr).actions{end+1} = 'NA';
        else
            vidanno(fr).actions{end+1} = astr(anno.people(i).attr(2, j));
        end
    end
end

for i = 1:length(anno.people)
    for j = i+1:length(anno.people)
        frames = intersect(anno.people(i).time, anno.people(j).time);
        
        iid = get_interaction_idx(i, j, length(anno.people));
        
        for t = 1:length(frames)
            fr = frames(t);
            
            if(anno.interaction(iid, fr) <= 1)
                continue;
            else
                vidanno(fr).interactions{end+1} = istr(anno.interaction(iid, fr));
            end
            
            
            bb1 = anno.people(i).sbbs(:, anno.people(i).time == fr);
            bb2 = anno.people(j).sbbs(:, anno.people(j).time == fr);
            
            vidanno(fr).ilines(:, end+1) = [bb1(1) + bb1(3) / 2; ...
                                            bb1(2) + bb1(4); ...
                                            bb2(1) + bb2(3) / 2; ...
                                            bb2(2) + bb2(4)];
        end
    end
end

end



function idx = get_interaction_idx(a1, a2, na)

assert(a1 > 0);
assert(a2 > 0);
assert(a1 <= na);
assert(a2 <= na);
assert(a1 ~= a2);

idx = get_pair_idx(min(a1, a2), max(a1, a2) - 1, na - 1);

end

function idx = get_pair_idx(min_idx, max_idx, num_label)
idx = (num_label * (num_label + 1)) / 2 ...
    - ((num_label - min_idx + 1) .* (num_label - min_idx + 2)) ./ 2 ... 
    + (max_idx - min_idx) + 1;
end
