select concat("\"",a.id,"\",",
              "\"",aa_make.a_value,"\",",
              "\"",aa_model.a_value,"\",",
              "\"",aa_year.a_value,"\",",
              "\"",aa_body.a_value,"\",",
              "\"",aa_color.a_value,"\",",
              "\"",a.poster_type,"\",",
              "\"",a.images,"\"")
from ad a, 
     ad_attributes aa_make,
     ad_attributes aa_model,
     ad_attributes aa_year,
     ad_attributes aa_body,
     ad_attributes aa_color
where a.category_id = 174 and a.ad_state = 'ACTIVE'
  and a.id = aa_make.ad_id and aa_make.a_name = 'carmake_s' 
  and a.id = aa_model.ad_id and aa_model.a_name = 'carmodel_s' 
  and a.id = aa_year.ad_id and aa_year.a_name = 'caryear_i'
  and a.id = aa_body.ad_id and aa_body.a_name = 'carbodytype_s'
  and a.id = aa_color.ad_id and aa_color.a_name = 'carcolor_s';

