# core
import os
import time

# dependencies
import fire
import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch

# src
from kymatio.torch import TimeFrequencyScattering1D


def resynthesise(
	target: torch.Tensor,
	jtfs: TimeFrequencyScattering1D,
	out_dir: str,
	learning_rate: float,
	bold_driver_accelerator: float = 0.98,
	bold_driver_accelerator_default: float = 1.1,
	bold_driver_brake: float = 0.55,
	idxs: torch.Tensor | None = None,
	n_iter: int = 100,
	sample_rate: int = 48000,
) -> None:
	N = target.shape[0]
	noise = torch.randn((N,), requires_grad=True)
	S_noise = jtfs(noise.cuda())
	err_history = []

	S_target = jtfs(target.cuda())
	curr_lr = learning_rate

	# Create an array filled with the float_value with the same shape as the input array
	if idxs is not None:
		new_array = torch.zeros(S_target.shape).cuda()
		new_array[:, idxs, :] = S_target[:, idxs, :]
		S_target = new_array

	os.makedirs(out_dir, exist_ok=True)

	for i in range(n_iter):
		# backward pass
		err = torch.norm(S_noise - S_target) / torch.norm(S_target)
		err_history.append(err)
		err.backward()

		# gradient descent
		delta_y = noise.grad
		if i % 1 == 0:
			print(f'{i}/{n_iter}: {err.cpu().detach().numpy():.4f}')
		if delta_y is not None:
			with torch.no_grad():
				noise_new = noise - curr_lr * delta_y
			noise_new.requires_grad = True

		print(f'learning rate: {curr_lr}')

		if err_history[i] > err_history[i - 1]:
			curr_lr *= bold_driver_brake
		else:
			curr_lr *= bold_driver_accelerator_default
			noise = noise_new

		if i >= 50:
			sf.write(
				os.path.join(out_dir, f'iter_{i:03}.wav'),
				noise.detach().cpu().numpy(),
				sample_rate,
				'PCM_32',
			)
		# forward pass
		S_noise = jtfs(noise.cuda())


def reconstruct(
	x: npt.NDArray[np.float32],
	jtfs: TimeFrequencyScattering1D,
	bold_driver_accelerator: float = 0.9,
	bold_driver_brake: float = 0.55,
	instance_name: str = '',
	j1: list[int] = [],
	learning_rate: float = 1.,
	n_iter: int = 150,
	output_dir: str = '',
	sample_rate: int = 48000,
) -> None:
	start = time.time()
	instance = f'{instance_name}_Q={jtfs.Q[0]}_J={jtfs.J}_Jfr={jtfs.J_fr}_accelerator={bold_driver_accelerator}'
	out_dir = os.path.join(output_dir, instance)
	os.makedirs(out_dir, exist_ok=True)
	target = torch.from_numpy(x).cuda()
	sf.write(os.path.join(out_dir, 'target.wav'), x, sample_rate, 'PCM_32')

	torch.manual_seed(0)

	S_target = jtfs(target.cuda())
	idxs = get_unique_j1(jtfs, S_target[0].mean(dim=-1))

	if j1:
		idxs = [x for i, x in enumerate(idxs) if i in j1]
	order1 = torch.tensor(np.where(np.isin(jtfs.meta()['order'], [0, 1]))[0])

	resynthesise(
		target,
		jtfs,
		bold_driver_accelerator=bold_driver_accelerator,
		bold_driver_brake=bold_driver_brake,
		idxs=None,
		learning_rate=learning_rate,
		out_dir=os.path.join(out_dir, 'all'),
		n_iter=n_iter,
		sample_rate=sample_rate,
	)

	for j, idx in enumerate(idxs):
		if not idx:
			continue
		print(f'Band {j}')
		curr_idxs = torch.cat([order1, torch.tensor(idx)])
		bin_dir = os.path.join(out_dir, f'{j}')
		resynthesise(target, jtfs, idxs=curr_idxs, learning_rate=learning_rate, n_iter=n_iter, out_dir=bin_dir)

	end = time.time()
	print(end - start)


def get_unique_j1(jtfs: TimeFrequencyScattering1D, Sx: torch.Tensor) -> list[list[int]]:
	order01 = np.where(np.isin(jtfs.meta()['order'], [0, 1]))
	Sx_sorted = Sx.argsort()
	sort_desc = torch.flip(Sx_sorted[torch.from_numpy(~np.isin(Sx_sorted.cpu(), order01))], dims=(0, )).cpu()
	js = jtfs.meta()['j']
	j1_to_idx: dict[int, list[int]] = {i: [] for i in np.unique(js[:, 0]) if ~np.isnan(i)}
	for i in sort_desc:
		j1 = js[i][0]
		if ~np.isnan(j1):
			j1_to_idx[j1].append(int(i))
	return [v for v in j1_to_idx.values()]


def run_resynth(
	audio_dir: str = '/Users/lewiswolstanholme/Desktop/hearing-from-within-a-sound/python/test/audio',
	input_len: int | None = None,
	j1: list[int] = [],
	learning_rate: float = 1.,
	max_length: float = 15.,
	n_iter: int = 150,
	output_dir: str = os.path.join(os.getcwd(), 'out/'),
) -> None:
	# handle input errors
	if not audio_dir:
		raise ValueError('Directory of audio files must be specified: `--audio_dir </absolute/path/to/audio/files/>`')

	# assert len(audio_files) == len(start)
	audio_files = os.listdir(audio_dir)
	for i, audio_file in enumerate(audio_files):
		# import x, reduce type and convert to mono
		x, sample_rate = sf.read(os.path.join(audio_dir, audio_file))
		x = np.float32(x)
		if x.shape[1] != 1:
			x = x.sum(axis=1) / x.shape[1]
		if x.shape[-1] > max_length * sample_rate:
			raise ValueError(f'{audio_file} exceeds maximum allowed sample length.')

		print(f'Currently resynthesising: {audio_file}')
		x_input = x[:sample_rate * input_len] if input_len else x
		N = x_input.shape[0]
		jtfs = TimeFrequencyScattering1D(
			J=13,
			shape=(N,),
			Q=(12, 1),
			Q_fr=1,
			J_fr=5,
			max_pad_factor=1,
			max_pad_factor_fr=1,
			average_fr=False,
			oversampling=0,
			oversampling_fr=0,
			normalize='l1-energy',
			analytic=True,
		).cuda()

		reconstruct(
			x_input,
			jtfs,
			instance_name=os.path.splitext(os.path.basename(audio_file))[0],
			j1=j1,
			learning_rate=learning_rate,
			n_iter=n_iter,
			output_dir=output_dir,
			sample_rate=sample_rate,
		)


if __name__ == '__main__':
	fire.Fire(run_resynth)
