function [pts, segments] = pick_fault_trace_continuous()
% Left click: add / drag / insert
% Right click: delete nearest point / undo last
% Enter: switch edit mode / finish

pts = zeros(0,2);
segments = zeros(0,4);

%% Load grid
[fname, fpath] = uigetfile('*.grd', 'Select InSAR .grd file');
if isequal(fname, 0), return; end
[x, y, z] = grdread2(fullfile(fpath, fname));

%% Optional import
if strcmp(questdlg('Import an existing fault trace?', 'Import Trace', 'Yes', 'No', 'No'), 'Yes')
    [tname, tpath] = uigetfile('*.txt', 'Select trace txt file');
    if ~isequal(tname, 0)
        pts = load_trace_points(fullfile(tpath, tname));
    end
end

%% Figure
fig = figure('Name', 'InSAR Fault Trace Picker', 'NumberTitle', 'off');
ax = axes('Parent', fig);
hImg = imagesc(ax, x, y, z);
set(hImg, 'AlphaData', ~isnan(z));   % NaN 透明
axis(ax, 'xy');
axis(ax, 'equal');
colormap(ax, jet);
colorbar(ax);
set(fig, 'Color', [0.85 0.85 0.85]);
set(ax,  'Color', [0.85 0.85 0.85], 'Layer', 'top');
hold(ax, 'on');
xlabel(ax, 'Longitude');
ylabel(ax, 'Latitude');
title(ax, {'Left click: add / drag / insert', ...
           'Right click: delete / undo', ...
           'Enter: edit mode / finish'}, ...
           'FontSize', 11);

%% State
mode = "pick";
dragging = false;
dragIdx = [];
tolP = 0.02;
tolS = 0.02;

ptsH = gobjects(0);
segH = gobjects(0);
txtH = gobjects(0);
set(fig, 'WindowButtonDownFcn',   @onDown);
set(fig, 'WindowButtonMotionFcn', @onMove);
set(fig, 'WindowButtonUpFcn',     @onUp);
set(fig, 'KeyPressFcn',           @onKey);

redraw();
uiwait(fig);

%% Output
if size(pts,1) >= 2
    segments = [pts(1:end-1,:) pts(2:end,:)];
end

if ishandle(fig), close(fig); end
% --- Ask user whether to save ---
choice = questdlg('Save fault trace to .txt file?', ...
                  'Save Output', 'Yes', 'No', 'Yes');

if strcmp(choice, 'Yes')

    % --- Ask for output format ---
    fmt_choice = questdlg( ...
        ['Choose output format:' newline newline ...
         'Segment rows:  lon1 lat1 lon2 lat2' newline ...
         '               lon3 lat3 lon4 lat4' newline newline ...
         'Point rows:    lon1 lat1' newline ...
         '               lon2 lat2' newline ...
         '               lon3 lat3' newline ...
         '               lon4 lat4'], ...
        'Output Format', ...
        'Segment rows (endpoint pairs)', ...
        'Point rows (one point per line)', ...
        'Segment rows (endpoint pairs)');   % default

    if isempty(fmt_choice)
        fprintf('Save cancelled.\n');
        return;
    end

    [out_fname, out_path] = uiputfile('*.txt', 'Save fault trace as', ...
                                      'fault_trace.txt');
    if isequal(out_fname, 0)
        fprintf('Save cancelled.\n');
    else
        out_file = fullfile(out_path, out_fname);
        fid = fopen(out_file, 'w');
        if fid == -1
            error('Could not open file for writing: %s', out_file);
        end

        if strcmp(fmt_choice, 'Segment rows (endpoint pairs)')
            % Format: lon1 lat1 lon2 lat2  (one segment per line)
            for i = 1:size(segments,1)
                fprintf(fid, '%.6f  %.6f  %.6f  %.6f\n', ...
                        segments(i,1), segments(i,2), ...
                        segments(i,3), segments(i,4));
            end
            fprintf('Format: segment rows (lon1 lat1 lon2 lat2)\n');

        else
            % Format: lon lat  (one point per line, two lines per segment)
            fprintf(fid, '%.6f  %.6f\n', segments(1,1), segments(1,2));
            for i = 1:size(segments,1)
                fprintf(fid, '%.6f  %.6f\n', segments(i,3), segments(i,4));
            end
            fprintf('Format: point rows (lon lat, one per line)\n');
        end

        fclose(fid);
        fprintf('Fault trace saved to: %s\n', out_file);
    end
else
    fprintf('Output not saved.\n');
end
%% ---------------- nested functions ----------------
    function onDown(~, ~)
        cp = ax.CurrentPoint;
        cx = cp(1,1);
        cy = cp(1,2);

        xl = xlim(ax);
        yl = ylim(ax);

        if cx < xl(1) || cx > xl(2) || cy < yl(1) || cy > yl(2)
            return;
        end

        if strcmp(fig.SelectionType, 'alt')   % right click
            idx = nearestPoint(cx, cy);
            if ~isempty(idx)
                pts(idx,:) = [];
            else
                pts(end,:) = [];
            end
            redraw();
            return;
        end

        % left click
        idx = nearestPoint(cx, cy);
        if ~isempty(idx)
            dragIdx = idx;
            dragging = true;
            return;
        end

        if size(pts,1) >= 2
            idx = nearestSegment(cx, cy);
            if ~isempty(idx)
                [px, py] = projectToSegment(cx, cy, pts(idx,:), pts(idx+1,:));
                pts = [pts(1:idx,:); px py; pts(idx+1:end,:)];
                dragIdx = idx + 1;
                dragging = true;
                redraw();
                return;
            end
        end

        pts(end+1,:) = [cx, cy];
        dragIdx = size(pts,1);
        dragging = true;
        redraw();
    end

    function onMove(~, ~)
        if ~dragging || isempty(dragIdx), return; end
        cp = ax.CurrentPoint;
        pts(dragIdx,:) = [cp(1,1), cp(1,2)];
        redraw();
    end

    function onUp(~, ~)
        dragging = false;
        dragIdx = [];
    end

    function onKey(~, evt)
        switch lower(evt.Key)
            case {'return','enter'}
                if mode == "pick"
                    if size(pts,1) < 2, return; end
                    mode = "edit";
                    title(ax, {'Edit mode: left click to drag/insert', ...
                               'Right click to delete / undo', ...
                               'Enter again to finish'}, 'FontSize', 11);
                else
                    uiresume(fig);
                end
            case {'backspace','delete'}
                if ~isempty(pts)
                    pts(end,:) = [];
                    redraw();
                end
            case {'escape','q'}
                uiresume(fig);
        end
    end

    function idx = nearestPoint(cx, cy)
        idx = [];
        if isempty(pts), return; end
        xr = max(diff(xlim(ax)), 1);
        yr = max(diff(ylim(ax)), 1);
        d = hypot((pts(:,1)-cx)/xr, (pts(:,2)-cy)/yr);
        [dmin, i0] = min(d);
        if dmin <= tolP, idx = i0; end
    end

    function idx = nearestSegment(cx, cy)
        idx = [];
        if size(pts,1) < 2, return; end
        xr = max(diff(xlim(ax)), 1);
        yr = max(diff(ylim(ax)), 1);

        bestD = inf;
        bestI = [];

        for i = 1:size(pts,1)-1
            [px, py, t] = projectToSegment(cx, cy, pts(i,:), pts(i+1,:));
            if t >= 0 && t <= 1
                d = hypot((px-cx)/xr, (py-cy)/yr);
                if d < bestD
                    bestD = d;
                    bestI = i;
                end
            end
        end

        if bestD <= tolS
            idx = bestI;
        end
    end

    function [px, py, t] = projectToSegment(cx, cy, p1, p2)
        v = p2 - p1;
        w = [cx, cy] - p1;
        den = dot(v, v);
        if den == 0
            t = 0;
            px = p1(1); py = p1(2);
            return;
        end
        t = dot(w, v) / den;
        t = max(0, min(1, t));
        p = p1 + t * v;
        px = p(1); py = p(2);
    end

    function redraw()
        xl = xlim(ax);
        yl = ylim(ax);

        if ~isempty(ptsH), delete(ptsH(isgraphics(ptsH))); end
        if ~isempty(segH), delete(segH(isgraphics(segH))); end
        if ~isempty(txtH), delete(txtH(isgraphics(txtH))); end

        n = size(pts,1);

        if n >= 2
            segH = gobjects(n-1,1);
            txtH = gobjects(n-1,1);
            for i = 1:n-1
                segH(i) = plot(ax, pts(i:i+1,1), pts(i:i+1,2), 'w-', 'LineWidth', 1.5);
                txtH(i) = text(ax, mean(pts(i:i+1,1)), mean(pts(i:i+1,2)), sprintf(' %d', i), ...
                    'Color', 'w', 'FontSize', 9);
            end
        end

        if ~isempty(pts)
            ptsH = plot(ax, pts(:,1), pts(:,2), 'ko', ...
                'MarkerFaceColor', 'y', 'MarkerSize', 6, 'LineWidth', 1.2);
        else
            ptsH = gobjects(0);
        end

        xlim(ax, xl);
        ylim(ax, yl);
        drawnow;
    end

    function ptsOut = load_trace_points(traceFile)
        M = readmatrix(traceFile);
        M = M(~all(isnan(M),2),:);

        if size(M,2) == 2
            ptsOut = M;
        elseif size(M,2) >= 4
            ptsOut = [M(1,1:2); M(:,3:4)];
        else
            error('Unsupported trace file format. Need 2 or 4 columns.');
        end
    end
end