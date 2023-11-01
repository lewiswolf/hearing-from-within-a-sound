# core
import os

# dependencies
import fire
import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch
from tqdm import tqdm

# src
from kymatio.torch import TimeFrequencyScattering1D


def resynthesise(
	target: torch.Tensor,
	jtfs: TimeFrequencyScattering1D,
	output_dir: str,
	learning_rate: float,
	bold_driver_accelerator: float = 1.08,
	bold_driver_brake: float = 0.6,
	idxs: torch.Tensor | None = None,
	instance_name: str = '',
	n_iter: int = 100,
	sample_rate: int = 48000,
) -> None:
	'''
	Perform JTFS resynthesis, parameterised by J bands.
	params:
		target						Target audio.
		jtfs						Initialised JTFS class.
		output_dir					Where to export the resynthesised material.
		learning_rate				Gradient descent update rate.
		bold_driver_accelerator		Coefficient to speed up gradient descent learning rate.
		bold_driver_brake			Coefficient to slow down gradient descent learning rate.
		idxs						J band locations.
		instance_name				Name of the target audio file.
		n_iter						Number of iterations of the gradient descent.
		sample_rate					Audio sample rate (hz).
	'''
	# configure output directory
	os.makedirs(output_dir, exist_ok=True)
	# initialis noise
	noise = torch.randn((target.shape[0],), requires_grad=True)
	S_target = jtfs(target.cuda())
	# create an array filled with the float_value with the same shape as the input array
	if idxs is not None:
		new_array = torch.zeros(S_target.shape).cuda()
		new_array[:, idxs, :] = S_target[:, idxs, :]
		S_target = new_array
	# initialise iteration history
	err_history = []
	current_learning_rate = learning_rate
	# gradient descent loop
	with tqdm(
		total=n_iter,
		bar_format='{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}'
		+ ', Loss: {postfix}   ',
		unit=' iterations',
	) as bar:
		for i in range(n_iter):
			# forward pass
			S_noise = jtfs(noise.cuda())
			err = torch.norm(S_noise - S_target) / torch.norm(S_target)
			err_history.append(err)
			# backward pass
			err.backward()
			if i > 0 and err_history[i] > err_history[i - 1]:
				current_learning_rate *= bold_driver_brake
			else:
				delta_y = noise.grad
				if delta_y is not None:
					with torch.no_grad():
						noise_new = noise - current_learning_rate * delta_y
					noise_new.requires_grad = True
					noise = noise_new
				current_learning_rate *= bold_driver_accelerator
			# export reconstructed audio
			if i >= 50:
				sf.write(
					os.path.join(output_dir, f'{i:03}_{instance_name}.wav'),
					noise.detach().cpu().numpy(),
					sample_rate,
					'PCM_32',
				)
			# loop stuff
			bar.postfix = err.cpu().detach().numpy()
			bar.update(1)


def reconstruct(
	x: npt.NDArray[np.float32],
	jtfs: TimeFrequencyScattering1D,
	instance_name: str = '',
	j1: list[int] = [],
	learning_rate: float = 1.,
	n_iter: int = 150,
	output_dir: str = '',
	sample_rate: int = 48000,
) -> None:
	'''
	Perform JTFS reconstructive synthesis, first as a complete transform, then on each J band.
	params:
		x							Target audio.
		jtfs						Initialised JTFS class.
		instance_name				Name of the target audio.
		j1							J bands to be resynthesised.
		learning_rate				Gradient descent update rate.
		n_iter						Amount of iterations the resynthesis algorithm performs.
		output_dir					Where the output audio files are saved.
		sample_rate					Audio sample rate (hz).
	'''
	# configure output directory
	instance = f'{instance_name}_Q={jtfs.Q[0]}_J={jtfs.J}_Jfr={jtfs.J_fr}'
	output_dir = os.path.join(output_dir, instance)
	os.makedirs(output_dir, exist_ok=True)
	# export a copy of the audio target
	sf.write(os.path.join(output_dir, 'target.wav'), x, sample_rate, 'PCM_32')
	# initialise target
	target = torch.from_numpy(x).cuda()
	torch.manual_seed(0)
	S_target = jtfs(target.cuda())
	order1 = torch.tensor(np.where(np.isin(jtfs.meta()['order'], [0, 1]))[0])
	# configure J bands
	idxs = get_unique_j1(jtfs, S_target[0].mean(dim=-1))
	idxs = [x for i, x in enumerate(idxs) if i in j1] if j1 else idxs
	# resynthesis loop
	with tqdm(
		total=len(idxs) + 1,
		bar_format='{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}   ',
		unit='band',
	) as bar:
		# create a resynthesis using all J bands
		resynthesise(
			target,
			jtfs,
			idxs=None,
			instance_name=instance_name,
			learning_rate=learning_rate,
			output_dir=os.path.join(output_dir, 'all'),
			n_iter=n_iter,
			sample_rate=sample_rate,
		)
		bar.update(1)
		# create a resynthesis for each J band individually
		for j, idx in enumerate(idxs):
			if idx:
				# resynthesis
				resynthesise(
					target,
					jtfs,
					bold_driver_accelerator=1.1,
					bold_driver_brake=0.55,
					idxs=torch.cat([order1, torch.tensor(idx)]),
					instance_name=instance_name,
					learning_rate=learning_rate,
					n_iter=n_iter,
					output_dir=os.path.join(output_dir, f'{j}'),
				)
			bar.update(1)


def get_unique_j1(jtfs: TimeFrequencyScattering1D, Sx: torch.Tensor) -> list[list[int]]:
	'''
	Retrieve the indexes of each J band.
	params:
		jtfs	Initialised JTFS class.
		Sx		Transform of the target audio.
	'''
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
	audio_dir: str = '',
	j1: list[int] = [],
	learning_rate: float = 1.,
	max_length: float = 15.,
	n_iter: int = 150,
	output_dir: str = os.path.join(os.getcwd(), 'out/'),
) -> None:
	'''
	Main routine for importing a directory of audio files and performing JTFS reconstructive synthesis.

	params:
		audio_dir 		Directory containing the input audiofiles.
		j1				J bands to be resynthesised.
		learning_rate	Gradient descent update rate.
		max_length		Maximum allowable length (seconds) of an input audio file. All audio files that exceed this duration
						will be trimmed.
		n_iter			Amount of iterations the resynthesis algorithm performs.
		output_dir		Where the output audio files are saved.
	'''
	# initialise input directory
	if not audio_dir:
		raise ValueError('Directory of audio files must be specified: `--audio_dir </absolute/path/to/audio/files/>`')
	audio_files = os.listdir(audio_dir)
	# loop over audio files
	for i, audio_file in enumerate(audio_files):
		# import x, convert to mono, and trim sample
		x, sample_rate = sf.read(os.path.join(audio_dir, audio_file))
		if x.shape[1] != 1:
			x = x.sum(axis=1) / x.shape[1]
		if x.shape[-1] > max_length * sample_rate:
			x = x[:max_length * sample_rate]
		# JTFS reconstructive synthesis
		print(f'Currently resynthesising: {audio_file}')
		reconstruct(
			x.astype(np.float32),
			TimeFrequencyScattering1D(
				J=13,
				shape=(x.shape[0],),
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
			).cuda(),
			instance_name=os.path.splitext(os.path.basename(audio_file))[0],
			j1=j1,
			learning_rate=learning_rate,
			n_iter=n_iter,
			output_dir=os.path.join(output_dir, os.path.basename(audio_dir)),
			sample_rate=sample_rate,
		)


if __name__ == '__main__':
	fire.Fire(run_resynth)
