_base_ = [
    './datasets/argo2-frustum.py',
    # '../_base_/nusc_frustum_with_aug_no_copy_paste_htc.py',
    '../../_base_/schedules/cosine_2x.py',
    '../../_base_/default_runtime.py',
]
#milestone mAP 70 NDS 72.2
#v10 + all_gelu all_l1_less_prev_cls_more_vel all_l1_points_512 dist_assign_no_barrier

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

seg_voxel_size = (0.2, 0.2, 0.2)
point_cloud_range = [-204.8, -204.8, -3.2, 204.8, 204.8, 3.2]
class_names = \
['Regular_vehicle',

 'Pedestrian',
 'Bicyclist',
 'Motorcyclist',
 'Wheeled_rider',

 'Bollard',
 'Construction_cone',
 'Sign',
 'Construction_barrel',
 'Stop_sign',
 'Mobile_pedestrian_crossing_sign',

 'Large_vehicle',
 'Bus',
 'Box_truck',
 'Truck',
 'Vehicular_trailer',
 'Truck_cab',
 'School_bus',
 'Articulated_bus',
 'Message_board_trailer',

 'Bicycle',
 'Motorcycle',
 'Wheeled_device',
 'Wheelchair',
 'Stroller',

 'Dog']
group1 = class_names[:1]
group2 = class_names[1:5]
group3 = class_names[5:11]
group4 = class_names[11:20]
group5 = class_names[20:25]
group6 = class_names[25:]
group_names=[group1, group2, group3, group4, group5, group6]
num_classes = len(class_names)

seg_score_thresh = [0.4, 0.25, 0.25, 0.25, 0.25, 0.25]
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5), len(group6)]

# tasks=[
#             dict(class_names=group1),
#             dict(class_names=group2),
#             dict(class_names=group3),
#             dict(class_names=group4),
#             dict(class_names=group5),
#             dict(class_names=group6),
#         ]
tasks=[dict(class_names=class_names),]

num_cams = 7
num_max_queries = 250 #unused
segmentor = dict(
    type='VoteSegmentor',
    tanh_dims=3,
    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=4,
        feat_channels=[64, 64],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
    ),

    middle_encoder=dict(
        type='PseudoMiddleEncoderForSpconvFSD',
    ),

    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=[32, 2048, 2048],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        decoder_channels=((128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 0), (1, 0), (0, 0), (0, 1)),
    ),

    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=67,
        hidden_dims=[128, 128],
        num_classes=num_classes,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='naiveSyncBN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1.0, ] * num_classes + [0.1,], 
            loss_weight=3.0),
        loss_vote=dict(
            type='L1Loss',
            loss_weight=1.0),
    ),
    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # for training log
        class_names=class_names, # for training log
        group_names=group_names,
        group_lens=group_lens,
    ),
    test_cfg=dict(
        point_loss=True,
        score_thresh=(0.5, 0.2, 0.2), # for test, no use
        clustering_voxel_size=(0.5, 0.5, 6), # for test, no use
    )
)

model = dict(
    type='SingleFrustum',

    segmentor=segmentor,

    painting_segmentor=True,
    segmentor_painting_mlp=dict(
        in_channel=32,
        mlp_channel=[128, 67],
        norm_cfg=dict(type='LN', eps=1e-3),
        act='gelu',
    ),

    num_classes=num_classes,
    num_cams=num_cams,
    num_max_queries=num_max_queries,
    class_names=class_names,
    
    encode_label_only=False,
    encode_2d_mlp_cfg=dict(
        in_channel=32,
        mlp_channel=[128, 128],
        norm_cfg=dict(type='LN', eps=1e-3),
        act='gelu',
    ),
    frustum_sir=dict(
        type='SIR',
        num_blocks=3, #179
        in_channels=[71,] + [132, ] * 2,
        feat_channels=[[128, 128], ] * 3,
        rel_mlp_hidden_dims=[[16, 32],] * 3,
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        xyz_normalizer=[20, 20, 4],
        act='gelu',
        unique_once=True,
    ),

    frustum_obj_head=dict(
        type='FrustumClusterHead',
        num_classes=num_classes,
        num_objs=num_max_queries,
        bbox_coder=dict(
            type='BasePointBBoxCoder',
            code_size=8,
        ),
        assigner=dict(
            type='FrustumAssigner',
            num_cams=num_cams,
            assigner_2d=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            assigner_3d=dict(
                type='PointInBoxAssigner',
            ),
            ignore_bev_dist=[8, 8, 10, 4, 4, 4],
            class_names=class_names,
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.0,
            alpha=0.25,
            loss_weight=4.0),
        loss_center=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
        loss_size=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
        loss_rot=dict(type='SmoothL1Loss', loss_weight=0.1, beta=0.1),
        in_channel=128 * 3 * 2 + 128,
        shared_mlp_dims=[1024, 1024],
        train_cfg=dict(),
        test_cfg=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.35,
            score_thr=0.01,
            min_bbox_size=0,
            max_num=500,  #6 * 83 < 500
        ),
        norm_cfg=dict(type='LN'),
        tasks=tasks,
        class_names=class_names,
        common_attrs=dict(
            center=(3, 2, 128), 
            dim=(3, 2, 128), 
            rot=(2, 2, 128), # (out_dim, num_layers, hidden_dim)
        ),
        num_cls_layer=2,
        cls_hidden_dim=128,
        separate_head=dict(
            type='FSDSeparateHead',
            norm_cfg=dict(type='LN'),
            act='gelu',
        ),
        as_rpn=False,
    ),
    ############ Multi stage ##############
    mlp_cfg=dict(
        embed_dims=1024,
        norm_cfg=dict(type='LN', eps=1e-3),
        act='gelu',
        lidar_img_input_dim=128 * 3 * 2 + 128, 
        lidar_input_dim=128 * 3 * 2,
    ),
    bbox_coder=dict(
            type='BasePointBBoxCoder',
            code_size=8,
    ),
    roi_extractor=dict(
                    type='DynamicPointROIExtractor',
                    extra_wlh=[1.0, 1.0, 1.0],
                    max_inbox_point=512,
                    debug=False,
                ),
    single_refine_sir_layer=dict(
                    type='FullySparseBboxHead',
                    num_classes=10,
                    num_blocks=3,
                    in_channels=[67+32+4+13, 130+13+2, 130+13+2], 
                    feat_channels=[[128, 128], ] * 3,
                    with_distance=False,
                    with_cluster_center=False,
                    with_rel_mlp=True,
                    rel_mlp_hidden_dims=[[16, 32],] * 3,
                    rel_mlp_in_channels=[13, ] * 3,
                    reg_mlp=[512, 512],
                    cls_mlp=[512, 512],
                    mode='max',
                    xyz_normalizer=[20, 20, 4],
                    cat_voxel_feats=True,
                    pos_fusion='mul',
                    fusion='cat',
                    act='gelu',
                    geo_input=True,
                    use_middle_cluster_feature=True,
                    norm_cfg=dict(type='LN', eps=1e-3),
                    unique_once=True,
                ),
    refined_obj_head=[
        dict(
            type='FrustumClusterHead',
            num_classes=num_classes,
            num_objs=num_max_queries,
            bbox_coder=dict(
                type='BasePointBBoxCoder',
                code_size=8,
            ),
            assigner=dict(
                type='FrustumAssigner',
                num_cams=num_cams,
                assigner_2d=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                assigner_3d=dict(
                    type='PointInBoxAssigner',
                    extra_height = 0.0
                ),
                class_names=class_names,
                tasks=tasks,
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=1.0,
                alpha=0.25,
                loss_weight=4.0),
            loss_center=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
            loss_size=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
            loss_rot=dict(type='SmoothL1Loss', loss_weight=0.1, beta=0.1),
            in_channel=1024,
            shared_mlp_dims=[1024, 1024],

            test_cfg=dict(
                use_rotate_nms=True,
                nms_pre=-1,
                nms_thr=0.35,
                score_thr=0.01,
                min_bbox_size=0,
                max_num=500,  #6 * 83 < 500
            ),
            norm_cfg=dict(type='LN'),
            tasks=tasks,
            class_names=class_names,
            common_attrs=dict(
                center=(3, 2, 128), 
                dim=(3, 2, 128), 
                rot=(2, 2, 128), # (out_dim, num_layers, hidden_dim)
            ),
            num_cls_layer=2,
            cls_hidden_dim=128,
            separate_head=dict(
                type='FSDSeparateHead',
                norm_cfg=dict(type='LN'),
                act='gelu',
            ),
            as_rpn=False,
        ),
    ],
    refine_encode_2d_mlp_cfg=dict(
        in_channel=32,
        mlp_channel=[32, 32],
        norm_cfg=dict(type='LN', eps=1e-3),
        act='gelu',
    ),
    #######################################
    backbone=dict(
        type='SIR',
        num_blocks=3,
        in_channels=[243 - 64,] + [132, ] * 2,
        feat_channels=[[128, 128], ] * 3,
        rel_mlp_hidden_dims=[[16, 32],] * 3,
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        xyz_normalizer=[20, 20, 4],
        act='gelu',
        unique_once=True,
    ),
    bbox_head=dict(
        type='SparseClusterHeadV2',
        num_classes=num_classes,
        bbox_coder=dict(type='BasePointBBoxCoder', code_size=8),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.0,
            alpha=0.25,
            loss_weight=4.0),
        loss_center=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
        loss_size=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
        loss_rot=dict(type='SmoothL1Loss', loss_weight=0.1, beta=0.1),
        in_channel=128 * 3 * 2,
        shared_mlp_dims=[1024, 1024],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type='LN'),
        tasks=tasks, #single-tasks
        class_names=class_names,
        common_attrs=dict(
            center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128),  # (out_dim, num_layers, hidden_dim)
        ),
        num_cls_layer=2,
        cls_hidden_dim=128,
        separate_head=dict(
            type='FSDSeparateHead',
            norm_cfg=dict(type='LN'),
            act='gelu',
        ),

    ),
    
    train_cfg=dict(
        score_thresh=seg_score_thresh,
        class_names=class_names, 
        sync_reg_avg_factor=True,
        pre_voxelization_size=(0.1, 0.1, 0.1),
        group_sample=True,
        group_names=group_names,
        offset_weight='max',
        group_lens=group_lens,
    ),
    test_cfg=dict(
        score_thresh=seg_score_thresh,
        class_names=class_names, 
        pre_voxelization_size=(0.1, 0.1, 0.1),
        group_sample=True,
        group_names=group_names,
        offset_weight='max',
        group_lens=group_lens,
        use_rotate_nms=True,
        nms_pre=-1,
        nms_thr=0.25, # from 0.25 to 0.7 for retest
        score_thr=0.1, 
        min_bbox_size=0,
        max_num=500,
    ),
    cluster_assigner=dict(
        cluster_voxel_size = [
            (0.3, 0.3, 6.4),
            (0.05, 0.05, 6.4),
            (0.08, 0.08, 6.4),
            (0.5, 0.5, 6.4),
            (0.1, 0.1, 6.4),
            (0.08, 0.08, 6.4),
        ],
        min_points=2,
        point_cloud_range=point_cloud_range,
        connected_dist=[0.6, 0.1, 0.15, 1.0, 0.2, 0.15],
        class_names=class_names,
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=6)
evaluation = dict(interval=6)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            load_interval=1)
    ),
)
log_config=dict(
    interval=20,
)

load_from='/mnt/weka/scratch/yingyan.li/repo/frustum-query-fusion/ckpt/argo_os_sp2_full12e_02.pth'
lr=1e-5
# custom_hooks = [
#     dict(type='DisableAugmentationHook', num_last_epochs=3, skip_type_keys=('MyObjectSample'), dataset_wrap=True),
#     # dict(type='EnableFSDDetectionHookIter', enable_after_iter=4000, threshold_buffer=0.3, buffer_iter=8000) 
# ]
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'segmentor.backbone': dict(lr_mult=0.2),
            'segmentor.voxel_encoder': dict(lr_mult=0.2),
        }),
)

num_dict = {136393:'Bicycle',
40137:'Bicyclist',
429857:'Bollard',
81933:'Bus',
1486565:'Pedestrian',
3795598:'Regular_vehicle',
79169:'Sign',
142649:'Stop_sign',
36656:'Vehicular_trailer',
3181:'Wheelchair',
78851:'Box_truck',
176099:'Construction_cone',
109464:'Large_vehicle',
38899:'Motorcycle',
13442:'Motorcyclist',
52754:'Truck',
75804:'Wheeled_device',
11174:'Dog',
4069:'Mobile_pedestrian_crossing_sign',
121998:'Construction_barrel',
17116:'Truck_cab',
10905:'School_bus',
8170:'Stroller',
10176:'Articulated_bus',
8714:'Wheeled_rider',
1119:'Official_signaler',
1224:'Railed_vehicle',
914:'Message_board_trailer',
156:'Traffic_light_trailer',
158:'Animal',}