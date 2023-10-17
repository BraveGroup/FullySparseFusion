_base_ = [
    '../_base_/datasets/nuscenes_dataloader.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py',
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

seg_voxel_size = (0.2, 0.2, 0.2)
point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
sparse_shape = [40, 512, 512]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

tasks = [
            dict(num_class=10, class_names=class_names),
        ]
num_classes = len(class_names)
group1 = ['car']
group2 = ['truck', 'construction_vehicle']
group3 = ['bus', 'trailer']
group4 = ['barrier']
group5 = ['motorcycle', 'bicycle']
group6 = ['pedestrian', 'traffic_cone']
group_names=[group1, group2, group3, group4, group5, group6]

seg_score_thresh = [0.1, ] * 6
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5), len(group6)]

segmentor = dict(
    type='VoteSegmentor',
    tanh_dims=[],
    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
    ),

    middle_encoder=dict(
        type='PseudoMiddleEncoderForSpconvFSD',
    ),

    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128, 
        encoder_channels=((128, ), (128, 128, 128), (128, 128, 128), (256, 256, 256), (512, 512, 512)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        decoder_channels=((512, 512, 256), (256, 256, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)), 
    ),

    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=67 + 64,
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
            loss_weight=10.0),
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
)

model = dict(
    type='FSF',
    num_classes=num_classes,
    num_cams=6,
    class_names=class_names,

    #LiDAR Query Generation
    segmentor=segmentor,
    backbone=dict(
        type='SIR',
        num_blocks=3,
        in_channels=[116 + 64,] + [133, ] * 2,
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
        bbox_coder=dict(type='BasePointBBoxCoder', code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_center=dict(type='L1Loss', loss_weight=0.5),
        loss_size=dict(type='L1Loss', loss_weight=0.5),
        loss_rot=dict(type='L1Loss', loss_weight=0.2),
        loss_vel=dict(type='L1Loss', loss_weight=0.2),
        in_channel=128 * 3 * 2,
        shared_mlp_dims=[1024, 1024],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type='LN'),
        tasks=tasks,
        class_names=class_names,
        common_attrs=dict(
            center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128), vel=(2, 2, 128)  # (out_dim, num_layers, hidden_dim)
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
        sync_reg_avg_factor=True,
        pre_voxelization_size=(0.1, 0.1, 0.1),
        disable_pretrain=False,
        disable_pretrain_topks=[200, ] * 6,
        group_sample=True,
        offset_weight='max',
        group_lens=group_lens,
        class_names=class_names, 
        group_names=[group1, group2, group3, group4, group5, group6],
    ),
    test_cfg=dict(
        score_thresh=seg_score_thresh,
        pre_voxelization_size=(0.1, 0.1, 0.1),
        group_sample=True,
        offset_weight='max',
        group_lens=group_lens,
        class_names=class_names, 
        group_names=[group1, group2, group3, group4, group5, group6],
        use_rotate_nms=True,
        nms_pre=-1,
        nms_thr=0.25, # from 0.25 to 0.7 for retest
        score_thr=0.05, 
        min_bbox_size=0,
        max_num=500,
    ),
    cluster_assigner=dict(
        cluster_voxel_size = [
            (0.3, 0.3, 8),
            (0.3, 0.3, 8),
            (0.3, 0.3, 8),
            (0.1, 0.1, 8),
            (0.2, 0.2, 8),
            (0.05, 0.05, 8),
        ],
        min_points=2,
        point_cloud_range=point_cloud_range,
        connected_dist=[0.6, 0.6, 0.6, 0.2, 0.4, 0.1],
        class_names=class_names,
    ),

    #Camera Query Generation
    frustum_sir=dict(
        type='SIR',
        num_blocks=3,
        in_channels=[67 + 64 + 5,] + [133, ] * 2,
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
        bbox_coder=dict(
            type='BasePointBBoxCoder',
            code_size=10,
        ),
        assigner=dict(
            type='HybridAssigner',
            num_cams=6,
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
            class_names=class_names,
            tasks=tasks,
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_center=dict(type='L1Loss', loss_weight=0.5),
        loss_size=dict(type='L1Loss', loss_weight=0.5),
        loss_rot=dict(type='L1Loss', loss_weight=0.2),
        loss_vel=dict(type='L1Loss', loss_weight=0.2),
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
            rot=(2, 2, 128), 
            vel=(2, 2, 128), # (out_dim, num_layers, hidden_dim)
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

    #Query Refinement
    mlp_cfg=dict(
        embed_dims=1024,
        norm_cfg=dict(type='LN', eps=1e-3),
        act='gelu',
        lidar_img_input_dim=128 * 3 * 2 + 128, 
        lidar_input_dim=128 * 3 * 2,
    ),
    bbox_coder=dict(
            type='BasePointBBoxCoder',
            code_size=10,
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
                    in_channels=[67+5+13+32 + 64, 131+13+2, 131+13+2], 
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
            bbox_coder=dict(
                type='BasePointBBoxCoder',
                code_size=10,
            ),
            assigner=dict(
                type='FrustumAssigner',
                num_cams=6,
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
                assigner_dist=dict(
                    type='DistAssigner',
                    assign_tasks = [
                        dict(num_class=1, class_names=["car"]),
                        dict(num_class=1, class_names=["truck",]),
                        dict(num_class=1, class_names=["trailer"]),
                        dict(num_class=1, class_names=["bus"]),
                        dict(num_class=1, class_names=["construction_vehicle"]),
                        dict(num_class=1, class_names=["bicycle"]),
                        dict(num_class=1, class_names=["motorcycle"]),
                        dict(num_class=1, class_names=["pedestrian"]),
                        dict(num_class=1, class_names=["traffic_cone"]),
                        dict(num_class=1, class_names=["barrier"]),
                    ],
                    ##          Car    truck  trailer bus   cv     bicycle motorcycle  pedestrian traffic_cone barrier
                    max_dist = [[1.0], [1.0], [2.0], [4.0], [0.5], [0.5],  [0.5],      [0.5],     [0.5],       [0.0],],
                    class_names=class_names,
                ),
                class_names=class_names,
                tasks=tasks,
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=4.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_center=dict(type='L1Loss', loss_weight=0.5),
            loss_size=dict(type='L1Loss', loss_weight=0.5),
            loss_rot=dict(type='L1Loss', loss_weight=0.2),
            loss_vel=dict(type='L1Loss', loss_weight=0.2),
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
                rot=(2, 2, 128), 
                vel=(2, 2, 128), # (out_dim, num_layers, hidden_dim)
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
        in_channel=10, 
        mlp_channel=[32, 32],
        norm_cfg=dict(type='LN', eps=1e-3),
        act='gelu',
    ),
    
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=6)
evaluation = dict(interval=6)

log_config=dict(
    interval=20,
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'segmentor.backbone': dict(lr_mult=0.2),
            'segmentor.voxel_encoder': dict(lr_mult=0.2),
        }),
)

load_from='ckpt/fsd_pretrain_backbone.pth'