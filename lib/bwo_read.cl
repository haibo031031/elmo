float _clLMBWORead(__local float * cacher, int length_x, int offset_x, int offset_y, int cur_offset_x, int cur_offset_y){
	int t_lc_x = get_local_id(0); 
	int t_lc_y = get_local_id(1); 
	int d_lc_x = t_lc_x + offset_x + cur_offset_x; 
	int d_lc_y = t_lc_y + offset_y + cur_offset_y; 
	return cacher[d_lc_y * length_x + d_lc_x]; 
}
