"""Desktop controls for BOR execution, separate from 2D profiles."""
from PySide6.QtWidgets import QGroupBox, QFormLayout, QComboBox, QSpinBox
from ghost_backend.bor.options import validate_options


class BorOptionsWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__('BOR execution and resources', parent)
        form=QFormLayout(self)
        self.factorization=QComboBox()
        self.factorization.addItem('Automatic', 'auto')
        self.factorization.addItem('Dense LU', 'dense')
        self.factorization.addItem('Compressed assembly (experimental)', 'compressed')
        form.addRow('Factorization',self.factorization)
        self.batch=QSpinBox()
        self.batch.setRange(1,256)
        self.batch.setValue(64)
        form.addRow('Aspects per batch',self.batch)
        self.reuse=QComboBox()
        for label,value in [('Automatic','auto'),('Off','off'),('On','on')]:
            self.reuse.addItem(label,value)
        form.addRow('Incident basis reuse',self.reuse)
        self.quadrature=QComboBox()
        self.quadrature.addItem('Standard', 'off')
        self.quadrature.addItem('Compare refined integration', 'refine')
        self.quadrature.setToolTip('Runs twice on the same mesh with finer self/adjacent/junction integration. Checks complex fields at the requested angles and returns the refined result only if they agree. Adds solve time; does not certify the drawn shape.')
        form.addRow('Integration accuracy check',self.quadrature)
        self.storage=QSpinBox()
        self.storage.setRange(0,1048576)
        self.storage.setValue(0)
        self.storage.setSuffix(' MiB')
        self.storage.setSpecialValueText('Automatic')
        self.storage.setToolTip('Combined numeric operator and inverse storage across active modes, shared by the concurrently factored modes. Automatic sizes it from the solve memory limit. Workspaces and near quadrature require additional RAM.')
        form.addRow('Compressed storage cap',self.storage)
        self.tile=QSpinBox()
        # The out-of-domain minimum is only a UI sentinel; profiles store
        # the string 'auto', preserving all existing integer8..128 settings.
        self.tile.setRange(7,128)
        self.tile.setSpecialValueText('Automatic')
        self.tile.setValue(7)
        form.addRow('Compression tile size',self.tile)
        self.cache=QSpinBox()
        self.cache.setRange(0,4096)
        self.cache.setValue(16)
        self.cache.setSuffix(' MiB')
        self.cache.setToolTip('Shared cache for repeated compressed coefficient queries; 0 disables it. Additional to the compressed storage cap.')
        form.addRow('Coefficient tile cache',self.cache)
        self.storage.setEnabled(False)
        self.tile.setEnabled(False)
        self.cache.setEnabled(False)
        self.factorization.currentIndexChanged.connect(self._sync)

    def _sync(self):
        # Automatic may still resolve to the compressed path, so its controls stay live.
        compressed=self.factorization.currentData() in ('compressed','auto')
        self.storage.setEnabled(compressed)
        self.tile.setEnabled(compressed)
        self.cache.setEnabled(compressed)

    def set_value(self, raw):
        value=validate_options(raw)
        self.factorization.setCurrentIndex(self.factorization.findData(value['factorization']))
        self.batch.setValue(value['angle_batch_size'])
        self.reuse.setCurrentIndex(self.reuse.findData(value['rhs_compression']))
        self.storage.setValue(value['compressed_storage_mib'])
        self.tile.setValue(7 if value['compression_tile'] == 'auto' else value['compression_tile'])
        self.cache.setValue(value['tile_cache_mib'])
        self.quadrature.setCurrentIndex(self.quadrature.findData(value['quadrature_check']))
        self._extra_options = {key:value[key] for key in ('near_backend','stream_spill','far_compression','near_refinement')}
        self._sync()

    def value(self):
        return validate_options(dict(getattr(self, '_extra_options', {}), factorization=self.factorization.currentData(),
            angle_batch_size=self.batch.value(),rhs_compression=self.reuse.currentData(),
            compressed_storage_mib=self.storage.value(),
            compression_tile='auto' if self.tile.value() == 7 else self.tile.value(),
            tile_cache_mib=self.cache.value(),quadrature_check=self.quadrature.currentData()))
