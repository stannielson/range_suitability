"""
Title:          Range Suitability Model

Authors:        Stanton K. Nielson
                GIS Specialist
                Bureau of Land Management Wyoming
                High Desert District
                snielson@blm.gov

                Taylor Grysen
                Geologist
                Bureau of Land Management Nevada
                Humboldt River Field Office
                tgrysen@blm.gov

Date:           July 25, 2018

Version:        2.0

Description:    This tool identifies range suitability for livestock, combining
                buffered hydrography and classified clope to identify areas of
                suitability reduction for effective range management. The
                basis for this tool methodology derives from Holechek (1988),
                along with Oberlie and Bishop (2009), but with incorporation of
                linear regression analysis techniques to interpolate values
                within established intervals. The output data depicts minimum
                suitability reduction percentage for the area (except where
                suitability reduction is at 100%).

Citations:      Holechek, J. L. (1988). An approach for setting stocking rate
                for cattle grazing capacity for different percentages of slope.
                Rangelands, 10, 10-14.

                Holechek, J. L., Pieper, R. D., & Herbel, C. H. (5th ed.)
                (2004). Considerations Concerning Stocking Rate. In J. L.
                Holechek, R. D. Pieper, & C. H. Herbel, Range Management:
                Principals and Practices (pp. 216-260). Upper Saddle River, NJ:
                Pearson/Prentice Hall.

                Oberlie, D. L. & Bishop, J. A. (2009). Determining rangeland
                suitability for cattle grazing based on distance-to-water,
                terrain, and barriers-to-movement attributes. Unpublished
                master's thesis, Pennsylvania State University, State College,
                Pennsylvania.                
"""


import arcpy
import datetime
import os


class RangeSuitability(object):

    """
    The RangeSuitability object is an encapsulated construct that performs the
    necessary processes to produce range suitability data based on DEM and
    hydrology data, with ability exclude areas known to be unsuitable.  All of
    the functions are internal, and require no action from the user beyond
    specifying the initial parameters for initialization.  These parameters
    are:

    - input_dem:            the digital elevation model required to analyze
                            suitability based on slope.

    - process_workspace:    the geodatabase required to temporarily house
                            intermediary data.

    - suitability_output:   the output features for suitability reduction
                            analysis. Areas depict minimum suitability
                            reduction as a percentage (except where suitability
                            is at 100%).

    - hydrography_list:     list of hydrographic features required to analyze
                            suitability based on distance from water (accepts
                            points, lines, and polygons).  WARNING: larger
                            datasets will create excessive processing time.
                            Subsetting is recommended to manage processing.

    - reduction_interval    (default=10) interval in whole percentage points
                            for each level of suitability reduction within the
                            target spatial envelope.

    - exclusion_list:       (optional) list of features to exclude from
                            suitability analysis (polygons only).

    - project_area:         (optional) area to subset suitability reduction
                            (polygon only).  WARNING: larger datasets will
                            create excessive processing time.  Subsetting is
                            recommended to manage processing.
    """
    

    def __init__(self, input_dem, process_workspace, suitability_output,
                 hydrography_list, reduction_interval=10, exclusion_list=None,
                 project_area=None):

        # Variables derived from arguments
        self.input_dem = input_dem
        self.process_workspace = process_workspace
        self.suitability_output = suitability_output
        self.hydrography_list = hydrography_list
        self.reduction_interval = reduction_interval

        # Variables derived from validated arguments
        self.exclusion_list = self.__validate_exclusion_list(exclusion_list)
        self.project_area = self.__validate_project_area(project_area)

        # Timestamp creation for use in temporary filenames
        self.timestamp = self.__generate_timestamp()

        # Dictionary object mapping hydrologic suitability intervals
        self.hydrologic_suitability_map = self.__calc_hydro_suitability(
            self.reduction_interval)

        # List object mapping slope suitability intervals
        self.slope_suitability_map = self.__calc_slope_suitability(
            self.reduction_interval)

        # Dictionary object for raster characteristics
        self.raster_char = self.__retrieve_raster_info(self.input_dem)

        # Setting the geoprocessing workspace
        arcpy.env.workspace = self.process_workspace

        # Setting the hydrologic project area to account for full dynamics
        self.hydro_project_area = self.__generate_hydro_project_area(
            self.project_area, self.timestamp)

        # Creation of buffered hydrography based on distance suitability 
        self.buffered_hydrography = self.__hydro_analysis(
            hydro_file_list=self.hydrography_list,
            distance_map=self.hydrologic_suitability_map,
            project_area=self.project_area,
            hydro_project_area=self.hydro_project_area,
            timestamp=self.timestamp)

        # Projection of buffered hydrography
        self.projected_hydrography = self.__project_feature_class(
            target_features=self.buffered_hydrography,
            output_features='tmp_hyd_proj_{}'.format(self.timestamp),
            output_coord_sys=self.raster_char['Spatial Reference'])

        # Conversion of hydrography features to raster
        self.hydrography_class_raster = self.__convert_hydro_to_raster(
            input_features=self.projected_hydrography,
            input_field='reduction',
            output_raster='tmp_hyd_ras_{}'.format(self.timestamp),
            cell_size=self.raster_char['Cell Size'])

        # Classification of slope derived from elevation raster
        self.slope_class_raster = self.__elev_analysis(
            elev_raster=self.input_dem,
            class_map=self.slope_suitability_map,
            hydro_area=self.projected_hydrography,
            timestamp=self.timestamp)

        # Analysis of overall suitability (prior to area exclusion)
        self.inclusive_suitability = self.__suitability_analysis(
            input_slope=self.slope_class_raster,
            input_hydrography=self.hydrography_class_raster,
            timestamp=self.timestamp)

        # Convertion of suitability raster to polygon features
        self.incl_suitability_poly = self.__convert_suitability_to_polygons(
            input_raster=self.inclusive_suitability,
            output_polygons='tmp_sut_fet_{}'.format(self.timestamp))

        # Removal of exclusionary areas from suitability features
        self.exclusive_suitability = self.__remove_exclusionary_areas(
            target_features=self.incl_suitability_poly,
            exclusion_list=self.exclusion_list,
            workspace=self.process_workspace,
            timestamp=self.timestamp)

        # Creation of output features
        self.__finalize_output(source_features=self.exclusive_suitability,
                               destination_features=self.suitability_output)

        # Deletion of intermediary data
        self.__delete_temporary_data(criteria=self.timestamp)

        
    def __elev_analysis(self, elev_raster, class_map, hydro_area, timestamp):

        arcpy.AddMessage('Processing elevation data...')
        clipped_elev = 'tmp_elv_clp_{}'.format(timestamp)
        arcpy.Clip_management(in_raster=elev_raster,
                              rectangle='#',
                              out_raster=clipped_elev,
                              in_template_dataset=hydro_area,
                              clipping_geometry='ClippingGeometry')

        arcpy.AddMessage('Calculating slope...')
        slope_data = 'tmp_slp_{}'.format(timestamp)
        slope_layer = arcpy.sa.Slope(in_raster=clipped_elev,
                                     output_measurement='PERCENT_RISE',
                                     z_factor=1)
        slope_layer.save(slope_data)
        del slope_layer
        
        arcpy.AddMessage('Classifying slope...')
        class_slope = 'tmp_cls_slp_{}'.format(timestamp)
        class_remap = arcpy.sa.RemapRange(class_map)
        class_layer = arcpy.sa.Reclassify(in_raster=slope_data,
                                          reclass_field='Value',
                                          remap=class_remap)
        class_layer.save(class_slope)
        del class_layer
        
        return class_slope
    

    def __hydro_analysis(self, hydro_file_list, distance_map, project_area,
                         hydro_project_area, timestamp):

        if hydro_project_area:

            arcpy.AddMessage('Processing hydrography...')
            clipped_features_map = {
                i: (hydro_file_list[i], 'tmp_hyd_clp_{:0>4}_{}'.format(
                    i, timestamp)) for i in range(0, len(hydro_file_list))}

            for index, paths in clipped_features_map.iteritems():

                features, clipped_features = paths
                arcpy.Clip_analysis(in_features=features,
                                    clip_features=hydro_project_area,
                                    out_feature_class=clipped_features)

            hydro_file_list = [i[1] for i in clipped_features_map.values()]

        interval_classes = distance_map.keys()
        interval_classes.sort()
        buffer_features_list = list()

        arcpy.AddMessage('Buffering...')

        for interval in interval_classes:

            buffer_subfeatures_list = list()
            distance = distance_map[interval]
            arcpy.AddMessage('({} mi. interval)'.format(distance))
            buffer_features = 'tmp_hyd_buf_{:0>4}_{}'.format(interval,
                                                             timestamp)

            for features in hydro_file_list:

                buffer_subfeatures = 'tmp_hyd_buf_{:0>4}_{:0>4}_{}'.format(
                    hydro_file_list.index(features), interval, timestamp)

                arcpy.Buffer_analysis(
                    in_features=features,
                    out_feature_class=buffer_subfeatures,
                    buffer_distance_or_field='{} Miles'.format(distance),
                    dissolve_option='ALL')

                buffer_subfeatures_list.append(buffer_subfeatures)

            subfeatures_union = 'tmp_hyd_uni_{:0>4}_{}'.format(interval,
                                                               timestamp)

            arcpy.Union_analysis(in_features=buffer_subfeatures_list,
                                 out_feature_class=subfeatures_union,
                                 join_attributes='NO_FID')

            arcpy.Dissolve_management(in_features=subfeatures_union,
                                      out_feature_class=buffer_features)

            arcpy.AddField_management(in_table=buffer_features,
                                      field_name='reduction',
                                      field_type='LONG')

            arcpy.CalculateField_management(in_table=buffer_features,
                                            field='reduction',
                                            expression='{}'.format(interval),
                                            expression_type='PYTHON_9.3')

            buffer_features_list.append(buffer_features)

        buffer_features_list.sort()
        buffer_features_list.reverse()

        update_input_features = buffer_features_list[0]
        update_list = buffer_features_list[1:]
        combined_buffers = 'tmp_com_buf_{}'.format(timestamp)

        arcpy.AddMessage('Combining buffers...')
        for features in update_list:

            update_index = update_list.index(features)
            update_output = 'tmp_hyd_upd_{:0>4}_{}'.format(update_index,
                                                           timestamp)
            if update_index == len(update_list) - 1:
                update_output = combined_buffers
                
            arcpy.Update_analysis(in_features=update_input_features,
                                  update_features=features,
                                  out_feature_class=update_output)
            
            update_input_features = update_output

        if project_area:
            arcpy.AddMessage('Clipping to project area...')
            clipped_combined_buffers = 'tmp_com_buf_clp_{}'.format(timestamp)
            arcpy.Clip_analysis(in_features=combined_buffers,
                                clip_features=project_area,
                                out_feature_class=clipped_combined_buffers)

            return clipped_combined_buffers

        else:

            return combined_buffers
        
        return
        
        
    def __suitability_analysis(self, input_slope, input_hydrography, timestamp):

        arcpy.AddMessage('Analyzing slope and hydrography...')
        raw_suit_raster = 'tmp_sut_raw_{}'.format(timestamp)
        slope_object = arcpy.sa.Raster(input_slope)
        hydro_object = arcpy.sa.Raster(input_hydrography)
        suitability_object = slope_object + hydro_object
        suitability_object.save(raw_suit_raster)

        del slope_object
        del hydro_object
        del suitability_object

        arcpy.AddMessage('Normalizing suitability reduction...')
        suitability_raster = 'tmp_sut_{}'.format(timestamp)
        #normal_suit_object = arcpy.sa.SetNull(
        #    in_conditional_raster=raw_suit_raster,
        #    in_false_raster_or_constant=raw_suit_raster,
        #    where_clause='VALUE >= 100')
        normal_suit_object = arcpy.sa.Con(
            in_conditional_raster=raw_suit_raster,
            in_true_raster_or_constant=raw_suit_raster,
            in_false_raster_or_constant='100',
            where_clause='VALUE < 100')
        normal_suit_object.save(suitability_raster)

        del normal_suit_object
        
        return suitability_raster
    

    def __calc_hydro_suitability(self, reduc_interval):

        # Returns the minimum suitability reduction specified by Holechek
        # et al (2004)
        dist_interval_int = int(2 * float(reduc_interval))
        dist_intervals = [float(i + dist_interval_int) / 100
                          if i + dist_interval_int <= 200
                          else 2.0
                          for i in range(0, 200, dist_interval_int)]
        reduc_intervals = [i * reduc_interval
                           for i in range(0, len(dist_intervals))]

        return {i: dist_intervals[reduc_intervals.index(i)]
                for i in reduc_intervals}


    def __calc_slope_suitability(self, reduc_interval):

        # Returns slope suitability reduction extrapolated from methods
        # described by Oberlie and Bishop (2009)

        # Generate series of derivative lists of map values (rounding applied
        # to eliminate range gaps due to limits of binary fractions)
        reduc_list = [i if i < 100 else 100 for i in range(
            0, 100 + reduc_interval, reduc_interval)]
        interv_min_list = [round(0.54 * i + 6.001, 3) for i in reduc_list]
        interv_max_list = [i for i in interv_min_list] + [999999999]

        # Alignment of list values through value insertion
        reduc_list = [0] + reduc_list
        interv_min_list = [0] + interv_min_list

        # Combination of lists to produce slope suitability map
        interv_map = [[i,
                       interv_max_list[interv_min_list.index(i)],
                       reduc_list[interv_min_list.index(i)]]
                      for i in interv_min_list]
        
        return interv_map


    def __convert_hydro_to_raster(self, input_features, input_field,
                                     output_raster, cell_size):
        
        arcpy.AddMessage('Converting hydrography to raster surface...')
        arcpy.FeatureToRaster_conversion(in_features=input_features,
                                         field=input_field,
                                         out_raster=output_raster,
                                         cell_size=cell_size)
        return output_raster


    def __convert_suitability_to_polygons(self, input_raster, output_polygons):

        arcpy.AddMessage('Converting suitability surface to features...')
        arcpy.RasterToPolygon_conversion(in_raster=input_raster,
                                         raster_field='Value',
                                         out_polygon_features=output_polygons,
                                         simplify='SIMPLIFY')
        
        return output_polygons


    def __delete_temporary_data(self, criteria):

        arcpy.AddMessage('Deleting intermediary data...')
        feature_list = arcpy.ListFeatureClasses(wild_card='*')
        raster_list = arcpy.ListRasters(wild_card='*')
        data_list = feature_list + raster_list
        filtered_data = [i for i in data_list if criteria in i]

        for data in filtered_data:
            print('({} of {})'.format(
                filtered_data.index(data) + 1, len(filtered_data)))
            while arcpy.Exists(data):
                try:
                    arcpy.Delete_management(in_data=data)
                except:
                    pass

        return


    def __finalize_output(self, source_features, destination_features):

        arcpy.AddMessage('Generating output features...')
        arcpy.DeleteField_management(in_table=source_features,
                                     drop_field='Id')
        arcpy.AlterField_management(in_table=source_features,
                                    field='gridcode',
                                    new_field_name='REDUCTION')
        destination_path, feature_class = os.path.split(destination_features)
        arcpy.FeatureClassToFeatureClass_conversion(
            in_features=source_features,
            out_path=destination_path,
            out_name=feature_class)

        return    
        
        
    def __generate_hydro_project_area(self, project_area, timestamp):

        if project_area:
            
            hydro_project_area = 'tmp_hyd_prj_{}'.format(timestamp)
            arcpy.Buffer_analysis(
                in_features=project_area,
                out_feature_class=hydro_project_area,
                buffer_distance_or_field='2.0 Miles',
                dissolve_option='ALL')
            
            return hydro_project_area

        else:
            
            return None
            
            
    def __generate_timestamp(self):
        
        utc_time = datetime.datetime.utcnow()
        timestamp = str(hex(int(utc_time.strftime(
            '%Y%m%d%H%M%S'))).rstrip('L').lstrip('0x'))
        del utc_time
        return timestamp


    def __project_feature_class(self, target_features, output_features,
                                output_coord_sys):

        arcpy.AddMessage('Aligning spatial reference...')
        arcpy.Project_management(in_dataset=target_features,
                                 out_dataset=output_features,
                                 out_coor_system=output_coord_sys)
        return output_features


    def __retrieve_raster_info(self, raster_dataset):

        raster_object = arcpy.Raster(raster_dataset)
        cell_dimensions = [raster_object.meanCellWidth,
                           raster_object.meanCellHeight]
        return {'Cell Size':            min(cell_dimensions),
                'Spatial Reference':    raster_object.spatialReference}


    def __remove_exclusionary_areas(self, target_features, exclusion_list,
                                    workspace, timestamp):

        exclusive_features = 'tmp_exc_sut_{}'.format(timestamp)
        
        if exclusion_list:

            arcpy.AddMessage('Removing exclusionary areas...')
            
            if len(exclusion_list) == 1:
                exclusion_features = exclusion_list[0]
            else:
                exclusion_features = 'tmp_exc_fet_{}'.format(timestamp)
                arcpy.Union_analysis(
                    in_features=exclusion_list,
                    out_feature_class=os.path.join(
                        workspace, exclusion_features),
                    join_attributes='ONLY_FID')
            
            arcpy.Erase_analysis(
                in_features=target_features,
                erase_features=exclusion_features,
                out_feature_class=exclusive_features)
        else:
            exclusive_features = target_features

        return exclusive_features


    def __validate_exclusion_list(self, exclusion_list):

        if exclusion_list != []:
            return list(exclusion_list)
        else:
            return None


    def __validate_project_area(self, project_area):

        if project_area != '' and project_area:
            return project_area
        else:
            return None
            
        
if __name__ == '__main__':

    received_parameters = list(arcpy.GetParameter(i) for i in
                               range(arcpy.GetArgumentCount()))
    input_parameters = {
        'input_dem':            str(received_parameters[0]),
        'hydrography_list':     list({str(i) for i in received_parameters[1]}),
        'reduction_interval':   received_parameters[2],
        'process_workspace':    str(received_parameters[3]),
        'suitability_output':   str(received_parameters[4]),
        'exclusion_list':       list({str(i) for i in received_parameters[5]}),
        'project_area':         str(received_parameters[6])
        }
    
    range_suitability = RangeSuitability(**input_parameters)
