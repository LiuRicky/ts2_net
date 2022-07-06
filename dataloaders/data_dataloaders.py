import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
from dataloaders.dataloader_msrvttfull_retrieval import MSRVTTFULL_DataLoader
from dataloaders.dataloader_msvd_retrieval import MSVD_DataLoader
from dataloaders.dataloader_lsmdc_retrieval import LSMDC_DataLoader
from dataloaders.dataloader_activitynet_retrieval import ActivityNet_DataLoader
from dataloaders.dataloader_didemo_retrieval import DiDeMo_DataLoader
from dataloaders.dataloader_vatex_retrieval import VATEX_DataLoader
from dataloaders.dataloader_sthv2_classification import STHV2_DataLoader
from dataloaders.dataloader_ucf101_classification import UCF101_DataLoader
from dataloaders.dataloader_tgif_retrieval import TGIF_DataLoader

def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_msrvttfull_test(args, tokenizer, subset="test"):
    msrvttfull_testset = MSRVTTFULL_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    dataloader_msrvtt = DataLoader(
        msrvttfull_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvttfull_testset)

def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MSVD_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler

def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msvd_testset)


def dataloader_lsmdc_train(args, tokenizer):
    lsmdc_dataset = LSMDC_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler

def dataloader_lsmdc_test(args, tokenizer, subset="test"):
    lsmdc_testset = LSMDC_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        lsmdc_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(lsmdc_testset)


def dataloader_activity_train(args, tokenizer):
    activity_dataset = ActivityNet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activity_dataset), train_sampler

def dataloader_activity_test(args, tokenizer, subset="test"):
    activity_testset = ActivityNet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        activity_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(activity_testset)

def dataloader_tgif_train(args, tokenizer):
    tgif_dataset = TGIF_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(tgif_dataset)
    dataloader = DataLoader(
        tgif_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(tgif_dataset), train_sampler

def dataloader_tgif_test(args, tokenizer, subset="test"):
    tgif_testset = TGIF_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_tgif = DataLoader(
        tgif_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_tgif, len(tgif_testset)

def dataloader_didemo_train(args, tokenizer):
    didemo_dataset = DiDeMo_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(didemo_dataset), train_sampler

def dataloader_didemo_test(args, tokenizer, subset="test"):
    didemo_testset = DiDeMo_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_didemo = DataLoader(
        didemo_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_didemo, len(didemo_testset)

def dataloader_sthv2_train(args, tokenizer):
    sthv2_dataset = STHV2_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(sthv2_dataset)
    dataloader = DataLoader(
        sthv2_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(sthv2_dataset), train_sampler

def dataloader_sthv2_test(args, tokenizer, subset="test"):
    sthv2_testset = STHV2_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        sthv2_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(sthv2_testset)

def dataloader_vatex_train(args, tokenizer):
    vatex_dataset = VATEX_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_dataset)
    dataloader = DataLoader(
        vatex_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(vatex_dataset), train_sampler

def dataloader_vatex_test(args, tokenizer, subset="test"):
    vatex_testset = VATEX_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        vatex_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(vatex_testset)

def dataloader_ucf101_train(args, tokenizer):
    ucf101_dataset = UCF101_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(ucf101_dataset)
    dataloader = DataLoader(
        ucf101_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(ucf101_dataset), train_sampler

def dataloader_ucf101_test(args, tokenizer, subset="test"):
    ucf101_testset = UCF101_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        ucf101_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(ucf101_testset)

DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test, "test":None}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd_train, "val":dataloader_msvd_test, "test":dataloader_msvd_test}
DATALOADER_DICT["lsmdc"] = {"train":dataloader_lsmdc_train, "val":dataloader_lsmdc_test, "test":dataloader_lsmdc_test}
DATALOADER_DICT["activity"] = {"train":dataloader_activity_train, "val":dataloader_activity_test, "test":None}
DATALOADER_DICT["didemo"] = {"train":dataloader_didemo_train, "val":dataloader_didemo_test, "test":dataloader_didemo_test}
DATALOADER_DICT["sthv2"] = {"train":dataloader_sthv2_train, "val":dataloader_sthv2_test, "test":dataloader_sthv2_test}
DATALOADER_DICT["vatex"] = {"train":dataloader_vatex_train, "val":dataloader_vatex_test, "test":dataloader_vatex_test}
DATALOADER_DICT["ucf101"] = {"train":dataloader_ucf101_train, "val":dataloader_ucf101_test, "test":dataloader_ucf101_test}
DATALOADER_DICT["msrvttfull"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvttfull_test, "test":None}
DATALOADER_DICT["tgif"] = {"train":dataloader_tgif_train, "val":dataloader_tgif_test, "test":dataloader_tgif_test}