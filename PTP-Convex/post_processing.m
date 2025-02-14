function [val_mean, val_min, val_max] = post_processing(val_collector)
val_collector = val_collector(find(val_collector));
val_mean = mean(val_collector);
val_min = min(val_collector);
val_max = max(val_collector);
end