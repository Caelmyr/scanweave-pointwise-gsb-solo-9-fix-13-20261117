"""验证暗场/平场校正与几何处理的位置对应关系。

场景：暗场里有一条固定的列条纹（传感器固定模式噪声），
帧之间存在平移（模拟手动平移或自动配准写入的 matrix）。
修复前：先 warp 再校正 -> 校正模板与条纹错位，条纹复现，边界出现暗带。
修复后：先在原生分辨率校正再 warp -> 条纹始终被压下，无额外暗带。
"""
import sys
import os

_env_site = '/opt/pointwise-env/site-packages'
if os.path.isdir(_env_site) and _env_site not in sys.path:
    sys.path.insert(0, _env_site)
sys.path.insert(0, '/workspace')

import numpy as np
from scanweave.arrays import srgb_decode, srgb_encode
from scanweave.calibration import apply_calibration
from scanweave.geometry import warp, pixel_matrix
from scanweave.preparation import prepare


class FakeAssets:
    def __init__(self, library):
        self.library = dict(library)

    def field(self, asset_id):
        return self.library[asset_id].copy()

    def metadata(self, asset_id):
        image = self.library[asset_id]
        return {'width': image.shape[1], 'height': image.shape[0]}


def build_scene(h=120, w=160):
    rng = np.random.default_rng(7)
    # 在线性（传感器辐射）空间构造场景与暗场，再编码为 sRGB 显示值
    base_lin = np.full((h, w, 3), srgb_decode(np.array([[[0.45]]], np.float32)).item(), np.float32)
    base_lin += rng.normal(0, 0.002, (h, w, 3)).astype(np.float32)
    dark_lin = np.zeros((h, w, 3), np.float32)
    dark_lin[:, 40:43, :] = 0.05          # 固定列条纹（暗电流/列放大器）
    dark_lin[:, 100, :] = 0.03
    dark_lin[30, 70, :] = 0.8             # 热像素
    dark = srgb_encode(dark_lin)
    frame0 = srgb_encode(np.clip(base_lin + dark_lin, 0, 1))
    frame1 = srgb_encode(np.clip(np.roll(base_lin, 15, axis=1) + dark_lin, 0, 1))
    return srgb_encode(base_lin).astype(np.float32), dark.astype(np.float32), \
        frame0.astype(np.float32), frame1.astype(np.float32)


def stripe_energy(image, columns=(40, 41, 42, 100)):
    """固定条纹列相对相邻普通列的亮度抬升（传感器固定模式能量）。"""
    neighbors = [c for c in range(36, 46) if c not in columns] + [98, 99, 101, 102]
    stripe_mean = float(image[:, list(columns), :].mean())
    normal_mean = float(image[:, neighbors, :].mean())
    return max(0.0, stripe_mean - normal_mean)


def column_bump(image, columns, neighbors):
    return float(image[:, list(columns), :].mean() - image[:, list(neighbors), :].mean())


def main():
    h, w = 120, 160
    base, dark, frame0, frame1 = build_scene(h, w)
    assets = FakeAssets({'frame0': frame0, 'frame1': frame1, 'dark': dark})

    settings = {
        'enabled': True, 'dark': 'dark', 'flat': None,
        'dark_scale': 1.0, 'flat_strength': 1.0, 'hot_correct': False, 'hot_threshold': .05,
    }

    # --- 1. 纯校正语义：暗场必须在传感器原位置扣除 ---
    corrected = apply_calibration(frame0, settings, assets.field)[0]
    residual_stripe = stripe_energy(corrected)
    raw_stripe = stripe_energy(frame0)
    print(f'[1] 原始帧条纹能量={raw_stripe:.4f}  校正后={residual_stripe:.4f}')
    assert residual_stripe < raw_stripe * 0.25, '暗场未有效压下固定条纹'

    # --- 2. 模拟旧的错误顺序：先平移再按平移后图像做校正 ---
    matrix = pixel_matrix([[1, 0, 15 / w], [0, 1, 0], [0, 0, 1]], w, h)
    warped_raw = warp(frame1, matrix, (w, h))
    wrong_order = apply_calibration(warped_raw, settings, assets.field)[0]
    # 正确顺序：先校正，再平移
    right_order = warp(apply_calibration(frame1, settings, assets.field)[0], matrix, (w, h))
    # 平移后条纹应落在传感器原列 +15 的位置
    shifted_cols = [55, 56, 57, 115]
    shifted_neighbors = list(range(50, 62)) + [112, 113, 117, 118]
    e_wrong = column_bump(wrong_order, shifted_cols, shifted_neighbors)
    e_right = column_bump(right_order, shifted_cols, shifted_neighbors)
    print(f'[2] 错误顺序(先warp后校正)条纹能量={e_wrong:.4f}  '
          f'正确顺序(先校正后warp)={e_right:.4f}')
    assert e_right < e_wrong * 0.5, '正确顺序的条纹抑制应明显优于错误顺序'

    # --- 3. 端到端：prepare(aligned=True) 且帧带平移矩阵 ---
    project = {
        'reference': 'f' * 32,
        'settings': {
            'size': max(h, w),
            'calibration': settings,
            'lens': {'enabled': False},
        },
        'frames': [
            {'id': 'f' * 32, 'asset': 'frame0', 'matrix': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
             'mask': None, 'weight': 1., 'enabled': True, 'name': 'f0'},
            {'id': 'e' * 32, 'asset': 'frame1',
             'matrix': [[1., 0., 15 / w], [0., 1., 0.], [0., 0., 1.]],
             'mask': None, 'weight': 1., 'enabled': True, 'name': 'f1'},
        ],
    }
    prepared = prepare(project, assets, aligned=True)
    images = prepared['images']
    assert images[0].shape == (h, w, 3) and images[1].shape == (h, w, 3)

    # 平移帧的条纹区域（原传感器第40-42列，平移后应出现在约55-57列）
    shifted_cols = [55, 56, 57, 115]
    shifted_neighbors = list(range(50, 62)) + [112, 113, 117, 118]
    band_bump = column_bump(images[1][10:-10], shifted_cols, shifted_neighbors)
    print(f'[3] 平移后原条纹位置的亮度抬升={band_bump:.4f}（应接近0）')
    assert band_bump < 0.02, '平移后固定条纹重新出现'

    # --- 4. 暗带检查：warp 引入的零边界在校正后不能被放大成暗带 ---
    # 校正后再 warp，边界为 0；有效性遮罩应覆盖该区域
    validity = prepared['validities'][1]
    border = validity[:, :15]
    assert border.max() == 0, '平移空出的边界必须被有效性遮罩标记为无效'
    # 有效区域内第一列不应出现异常暗带
    first_valid_col = int(np.argmax(validity[h // 2] > 0.5))
    edge_value = float(images[1][10:-10, first_valid_col, :].mean())
    mid_value = float(images[1][10:-10, first_valid_col + 20, :].mean())
    print(f'[4] 有效区首列亮度={edge_value:.4f}  内部亮度={mid_value:.4f}')
    assert edge_value > mid_value * 0.8, '有效区域边缘出现额外暗带'

    # --- 5. 不启用校正时，输出应等于纯几何处理（校正不改变像素）---
    settings_off = dict(settings, enabled=False)
    project_off = dict(project, settings=dict(project['settings'], calibration=settings_off))
    prepared_off = prepare(project_off, assets, aligned=True)
    expected = warp(frame1, matrix, (w, h))
    assert np.allclose(prepared_off['images'][1], expected, atol=1e-6), '未启用校正时像素应与纯warp一致'
    assert prepared_off['calibration'][1] == {'enabled': False}
    print('[5] 未启用校正：输出与纯几何处理一致，报告 enabled=False')

    # --- 6. 原始素材未被修改 ---
    assert np.array_equal(assets.field('frame1'), frame1), '原始扫描被修改'
    assert np.array_equal(assets.field('dark'), dark), '暗场素材被修改'
    print('[6] 原始扫描与暗场素材保持不变')

    # --- 7. aligned=False（全景/drizzle/配准任务）行为保持：原生分辨率校正、无warp ---
    prepared_raw = prepare(project, assets, aligned=False)
    direct = apply_calibration(frame1, settings, assets.field)[0]
    assert np.allclose(prepared_raw['images'][1], direct, atol=1e-6), '非对齐路径应直接做原生校正'
    assert prepared_raw['calibration'][1]['enabled'] and prepared_raw['calibration'][1]['dark']
    print('[7] 非对齐路径：原生分辨率校正且不做几何变换，报告为真实校正结果')

    # --- 8. 平场校正同样必须在几何处理之前作用于原位置 ---
    rng = np.random.default_rng(11)
    flat_lin = np.tile(np.linspace(0.30, 0.42, w, dtype=np.float32)[None, :, None], (h, 1, 3))
    flat = srgb_encode(flat_lin)
    base_lin = np.full((h, w, 3), 0.18, np.float32) + rng.normal(0, .004, (h, w, 3)).astype(np.float32)
    # 内容平移15列，照明梯度仍固定在传感器列上
    f1_flat = srgb_encode(np.clip(np.roll(base_lin, 15, axis=1) * (flat_lin / flat_lin.mean()), 0, 1))
    assets_flat = FakeAssets({'frame0': frame0, 'frame1': f1_flat, 'flat': flat})
    settings_flat = {'enabled': True, 'dark': None, 'flat': 'flat', 'dark_scale': 1.0,
                     'flat_strength': 1.0, 'hot_correct': False, 'hot_threshold': .05}
    project_flat = dict(project, settings=dict(project['settings'], calibration=settings_flat))
    prepared_flat = prepare(project_flat, assets_flat, aligned=True)

    def col_residual(image):
        col_means = image[:, 30:120, :].mean(axis=(0, 2))
        return float(np.abs(col_means - col_means.mean()).max())

    right_residual = col_residual(prepared_flat['images'][1])
    # 对照旧的错误顺序
    matrix = pixel_matrix([[1, 0, 15 / w], [0, 1, 0], [0, 0, 1]], w, h)
    wrong = apply_calibration(warp(f1_flat, matrix, (w, h)), settings_flat, assets_flat.field)[0]
    wrong_residual = col_residual(wrong)
    print(f'[8] 平场列残差  正确顺序={right_residual:.5f}  错误顺序={wrong_residual:.5f}')
    assert right_residual < wrong_residual * 0.8, '平场在平移后未对齐原位置'
    assert np.array_equal(assets_flat.field('flat'), flat), '平场素材被修改'
    print('    平场素材保持不变')

    print('\n全部验证通过 ✓')


if __name__ == '__main__':
    main()
